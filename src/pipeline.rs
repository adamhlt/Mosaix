//! Capture → analysis → UI threads connected by channels; late frames are dropped.

use crate::analysis::{Analyzer, Report};
use crate::camera::SourceSpec;
use crate::capture::Params;
use crate::mosaic::Preview;
use crate::session::Session;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TrySendError};
use image::RgbImage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Message sent to the UI after each analysed frame.
pub struct UiUpdate {
    pub frame: Arc<RgbImage>,
    pub report: Report,
    /// New mosaic preview (only when it changed).
    pub preview: Option<Arc<Preview>>,
}

pub enum Event {
    Update(Box<UiUpdate>),
    /// End of stream (replay finished) or fatal error.
    Ended(String),
}

pub enum Control {
    Pause(bool),
    UndoLast,
    /// The row ends here.
    EndRow,
    /// The map is complete.
    Finish,
    /// Flip the row direction.
    Reverse,
    /// Resume mapping after a finish.
    Resume,
    Stop,
}

pub struct Pipeline {
    pub events: Receiver<Event>,
    ctrl: Sender<Control>,
    stop: Arc<AtomicBool>,
    session_dir: Arc<Mutex<Option<PathBuf>>>,
}

/// Non-blocking send into a capacity-1 channel: replaces the old item if it is still there.
fn send_latest<T>(tx: &Sender<T>, rx: &Receiver<T>, mut v: T) -> bool {
    loop {
        match tx.try_send(v) {
            Ok(()) => return true,
            Err(TrySendError::Full(back)) => {
                let _ = rx.try_recv(); // drop the old one
                v = back;
            }
            Err(TrySendError::Disconnected(_)) => return false,
        }
    }
}

impl Pipeline {
    /// Starts the threads. `wake` is called for every message to wake the UI.
    pub fn start(
        spec: SourceSpec,
        session_root: Option<PathBuf>,
        params: Params,
        preview_side: u32,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (frame_tx, frame_rx) = bounded::<Option<RgbImage>>(1);
        let (ev_tx, ev_rx) = bounded::<Event>(2);
        let (ctrl_tx, ctrl_rx) = unbounded::<Control>();
        let wake = Arc::new(wake);
        let stop = Arc::new(AtomicBool::new(false));
        let session_dir: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let session_dir_thread = session_dir.clone();

        // Capture thread: reads the source and publishes the latest frame.
        {
            let frame_tx = frame_tx.clone();
            let frame_rx = frame_rx.clone();
            let ev_tx = ev_tx.clone();
            let ev_rx = ev_rx.clone();
            let wake = wake.clone();
            let stop = stop.clone();
            std::thread::Builder::new()
                .name("capture".into())
                .spawn(move || {
                    let mut src = match spec.open() {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("opening source : {e:#}");
                            send_latest(&ev_tx, &ev_rx, Event::Ended(format!("{e:#}")));
                            wake();
                            return;
                        }
                    };
                    tracing::info!("source opened : {}", src.name());
                    while !stop.load(Ordering::Relaxed) {
                        match src.next_frame() {
                            Ok(Some(f)) => {
                                if !send_latest(&frame_tx, &frame_rx, Some(f)) {
                                    return; // analysis finished
                                }
                            }
                            Ok(None) => {
                                send_latest(&frame_tx, &frame_rx, None);
                                return;
                            }
                            Err(e) => {
                                tracing::error!("reading source : {e:#}");
                                send_latest(&ev_tx, &ev_rx, Event::Ended(format!("{e:#}")));
                                wake();
                                return;
                            }
                        }
                    }
                })
                .expect("failed to start the capture thread");
        }

        // Analysis thread
        {
            let ev_tx = ev_tx.clone();
            let ev_rx = ev_rx.clone();
            let wake = wake.clone();
            std::thread::Builder::new()
                .name("analyse".into())
                .spawn(move || {
                    let session = session_root.as_deref().and_then(|root| match Session::create(root) {
                        Ok(s) => {
                            tracing::info!("session: {}", s.dir().display());
                            if let Ok(mut g) = session_dir_thread.lock() {
                                *g = Some(s.dir().to_path_buf());
                            }
                            Some(s)
                        }
                        Err(e) => {
                            tracing::error!("session unavailable : {e:#}");
                            None
                        }
                    });
                    let mut an = Analyzer::new(params, session);
                    let t0 = Instant::now();
                    let mut paused = false;
                    let mut force_preview = false;
                    let mut last: Option<Report> = None;
                    while let Ok(msg) = frame_rx.recv() {
                        let Some(frame) = msg else {
                            send_latest(&ev_tx, &ev_rx, Event::Ended("end of stream".into()));
                            wake();
                            return;
                        };
                        while let Ok(c) = ctrl_rx.try_recv() {
                            match c {
                                Control::Pause(p) => paused = p,
                                Control::UndoLast => match an.undo_last() {
                                    Ok(_) => force_preview = true,
                                    Err(e) => tracing::error!("undo: {e:#}"),
                                },
                                Control::EndRow => an.end_row(),
                                Control::Finish => an.finish(),
                                Control::Reverse => an.reverse(),
                                Control::Resume => an.resume(),
                                Control::Stop => return,
                            }
                        }
                        let frame = Arc::new(frame);
                        if paused {
                            // Paused: keep showing the stream without analysing.
                            let mut report = last.clone().unwrap_or_else(|| empty_report(&frame, an.tile_count()));
                            report.captured = false;
                            report.tiles = an.tile_count();
                            send_latest(
                                &ev_tx,
                                &ev_rx,
                                Event::Update(Box::new(UiUpdate { frame, report, preview: None })),
                            );
                            wake();
                            continue;
                        }
                        match an.process(&frame, t0.elapsed().as_secs_f64()) {
                            Ok(report) => {
                                last = Some(report.clone());
                                let preview = if report.captured || force_preview {
                                    force_preview = false;
                                    an.mosaic.render(preview_side).map(Arc::new)
                                } else {
                                    None
                                };
                                send_latest(
                                    &ev_tx,
                                    &ev_rx,
                                    Event::Update(Box::new(UiUpdate { frame, report, preview })),
                                );
                                wake();
                            }
                            Err(e) => tracing::error!("analysis: {e:#}"),
                        }
                    }
                })
                .expect("failed to start the analysis thread");
        }

        Self { events: ev_rx, ctrl: ctrl_tx, stop, session_dir }
    }

    /// Folder of the session being recorded (None when saving is off or not created yet).
    pub fn session_dir(&self) -> Option<PathBuf> {
        self.session_dir.lock().ok().and_then(|g| g.clone())
    }

    pub fn send(&self, c: Control) {
        let _ = self.ctrl.send(c);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.ctrl.send(Control::Stop);
    }
}

fn empty_report(frame: &RgbImage, tiles: usize) -> Report {
    Report {
        state: crate::capture::State::Tracking,
        motion: None,
        pos: (0.0, 0.0),
        sharp: 0.0,
        sharp_rel: 1.0,
        stable_secs: 0.0,
        offset_frac: 0.0,
        tiles,
        frame_size: (frame.width(), frame.height()),
        captured: false,
        peak: 0.0,
        analysis_ms: 0.0,
        heading: Some((1.0, 0.0)),
        row_dir: 1.0,
        phase: crate::guidance::Phase::Row,
        rows_done: 0,
        finished: false,
    }
}
