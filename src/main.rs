//! Rune UI: start menu (source picker), live view with overlays,
//! side panel (mosaic, capture steps, metrics, details).
//!
//! `--camera <index> [--res WxH] [--yuyv]` skips the start menu.
//! Headless export of a saved session: `--export <session_dir> <out.png|jpg|tif> [--scale 0.5]
//! [--crop] [--no-refine]`.
//! UI testing without any source: `--fake stabilize|move|lost|closing|down|finished|dialog|update`
//! shows a frozen state.
// Release builds on Windows are a windowed app: no terminal window opens beside it. Debug builds
// keep their console for the logs.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod update;

use mosaix::analysis::Report;
use mosaix::camera::{Mode, PixelFormat, SourceSpec};
use mosaix::capture::{Params, State as CaptureState};
use mosaix::export::{export_session, ExportOptions, ExportResult, Format, Progress};
use mosaix::guidance::{Phase, Then};
use mosaix::mosaic::Preview;
use mosaix::pipeline::{Control, Event, Pipeline};
use rune::tok::{self, type_};
use rune::{
    canvas, col, el, row, spacer, text, AppSpec, Color, Cx, El, FileDialogOptions, Key, KeyEvent, Pixels, PopupMenu, Shape, SurfaceFit, Tx,
    Waker,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Name shown in the window title and the start menu (change it here only).
const APP_NAME: &str = "Mosaïx";

/// Max width of the image uploaded to the GPU for the live view.
const LIVE_MAX_W: u32 = 960;
const PANEL_W: f32 = 290.0;
const MOSAIC_BOX: (f32, f32) = (PANEL_W, 160.0);
const GAUGE_W: f32 = 130.0;
const RETICLE: f32 = 260.0;
/// Shown when the driver refuses to list its modes: common resolutions in the usual formats.
fn fallback_modes() -> Vec<Mode> {
    [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (800, 600), (640, 480)]
        .into_iter()
        .map(|(width, height)| Mode { width, height, formats: vec![(PixelFormat::Mjpeg, 30), (PixelFormat::Yuyv, 30)] })
        .collect()
}
const EXPORT_FORMATS: [(Format, &str, &str); 3] =
    [(Format::Png, "PNG", "png"), (Format::Jpeg, "JPEG", "jpg"), (Format::Tiff, "TIFF", "tif")];

// ---------------------------------------------------------------- colours
fn hex(s: &str) -> Color {
    rune::color::hex(s).unwrap_or(Color::WHITE)
}
fn amber() -> Color {
    hex("#f2c14e")
}
fn green() -> Color {
    hex("#5fd68a")
}
fn blue() -> Color {
    hex("#4c9bff")
}
fn red() -> Color {
    hex("#e24b4a")
}
fn dim() -> Color {
    hex("#8d8f98")
}
fn soft() -> Color {
    hex("#c9cbd2")
}
fn card() -> Color {
    hex("#1c1d22")
}
fn chip_bg() -> Color {
    Color::BLACK.alpha(0.72)
}

// ---------------------------------------------------------------- state
#[derive(Clone)]
enum CamList {
    Loading,
    Ready(Vec<(u32, String)>),
    Unavailable(String),
}

/// What the selected camera really offers (read from the driver on a worker thread).
#[derive(Clone)]
enum Modes {
    Idle,
    Loading,
    Ready(u32, Vec<Mode>),
    Failed(u32, String),
}

struct MenuState {
    cam: Option<u32>,
    modes: Arc<Mutex<Modes>>,
    /// Chosen resolution and format, always among the camera's real modes.
    res: Option<(u32, u32)>,
    format: Option<PixelFormat>,
    /// Folder in which every session (tiles + exported mosaics) is stored.
    save_dir: PathBuf,
    error: Option<String>,
}

/// A running export (background thread).
struct ExportJob {
    progress: Arc<Progress>,
    result: Arc<Mutex<Option<Result<ExportResult, String>>>>,
}

const EXPORT_SCALES: [(f32, &str); 3] = [(1.0, "100 %"), (0.5, "50 %"), (0.25, "25 %")];

#[derive(PartialEq)]
enum Screen {
    Menu,
    Running,
}

struct App {
    screen: Screen,
    menu: MenuState,
    cams: Arc<Mutex<CamList>>,
    waker: Option<Waker>,
    pipeline: Option<Pipeline>,
    live: Pixels,
    mosaic: Pixels,
    report: Option<Report>,
    preview: Option<Arc<Preview>>,
    paused: bool,
    ended: Option<String>,
    details_open: bool,
    export: Option<ExportJob>,
    export_msg: Option<(String, bool)>,
    export_crop: bool,
    export_scale: usize,
    export_format: usize,
    /// Session folder whose export options are being chosen (the options window is open).
    export_dialog: Option<PathBuf>,
    /// Checks GitHub for a newer release and installs it.
    updater: update::Updater,
    /// "Up to date" or a failed manual check, shown in the update bar until dismissed.
    update_note: Option<String>,
    /// A check asked for by hand is running: the bar says so instead of vanishing meanwhile.
    update_checking: bool,
    initial: Option<SourceSpec>,
    fps_frames: u32,
    fps_t0: f64,
    fps: f32,
}

fn parse_args() -> Option<SourceSpec> {
    let args = rune::extra_args();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let idx = get("--camera")?;
    let (width, height) = get("--res")
        .and_then(|r| r.split_once('x').map(|(a, b)| (a.parse().ok(), b.parse().ok())))
        .and_then(|(a, b)| a.zip(b))
        .unwrap_or((1920, 1080));
    Some(SourceSpec::Camera {
        index: idx.parse().unwrap_or(0),
        width,
        height,
        fps: 30,
        format: if args.iter().any(|a| a == "--yuyv") { PixelFormat::Yuyv } else { PixelFormat::Mjpeg },
    })
}

/// Frozen UI state for visual testing (`--fake <state>`): no threads, no source.
fn apply_fake(app: &mut App, kind: &str) {
    use mosaix::mosaic::Mosaic;
    let board = mosaix::camera::synthetic_board(1920, 1080);
    let view = image::imageops::crop_imm(&board, 320, 180, 1280, 720).to_image();
    app.live.set(view.width(), view.height(), rgb_to_rgba(&view));
    let mut mosaic = Mosaic::new();
    mosaic.add(&image::imageops::crop_imm(&board, 0, 0, 1280, 720).to_image(), (0.0, 0.0));
    mosaic.add(&image::imageops::crop_imm(&board, 640, 360, 1280, 720).to_image(), (870.0, 0.0));
    if let Some(pv) = mosaic.render(1600) {
        app.mosaic.set(pv.image.width(), pv.image.height(), pv.image.as_raw().clone());
        app.preview = Some(Arc::new(pv));
    }
    let (state, offset, stable, sharp_rel, phase, finished) = match kind {
        "move" => (CaptureState::Move, 0.42, 0.0, 0.92, Phase::Row, false),
        "lost" => (CaptureState::Lost, 0.30, 0.0, 0.41, Phase::Row, false),
        "closing" => (CaptureState::Stabilizing, 0.22, 0.12, 0.96, Phase::Closing(Then::StepDown), false),
        "down" => (CaptureState::Move, 0.12, 0.0, 0.96, Phase::Down, false),
        "finished" => (CaptureState::Tracking, 0.15, 0.0, 0.96, Phase::Row, true),
        _ => (CaptureState::Stabilizing, 0.68, 0.18, 0.96, Phase::Row, false),
    };
    let heading = if finished {
        None
    } else if phase == Phase::Down {
        Some((0.0, 1.0))
    } else {
        Some((1.0, 0.0))
    };
    app.report = Some(Report {
        state,
        motion: Some((0.4, -0.1)),
        pos: (870.0, 0.0),
        sharp: 3110.0,
        sharp_rel,
        stable_secs: stable,
        offset_frac: offset,
        tiles: 2,
        frame_size: (1280, 720),
        captured: false,
        peak: 0.3,
        analysis_ms: 2.8,
        heading,
        row_dir: 1.0,
        phase,
        rows_done: 1,
        finished,
    });
    if kind == "update" {
        let version = "9.9.0".to_string();
        let asset = update::Asset { name: "mosaix-windows-x64-portable.zip".into(), url: String::new(), size: 0, sha256: None };
        let release = update::Release { tag: format!("v{version}"), version, assets: vec![asset.clone()] };
        app.updater.state = update::State::Available(update::Offer { release, method: update::Method::Portable, asset: Some(asset) });
    }
    if kind == "dialog" {
        app.export_dialog = Some(PathBuf::from("sessions/session-1794758298"));
    }
    app.screen = Screen::Running;
}

fn state() -> App {
    let spec = parse_args();
    let fake = rune::extra_args().iter().position(|a| a == "--fake").and_then(|i| rune::extra_args().get(i + 1).cloned());
    let mut app = App {
        screen: Screen::Menu,
        menu: MenuState {
            cam: None,
            modes: Arc::new(Mutex::new(Modes::Idle)),
            res: None,
            format: None,
            save_dir: PathBuf::from("sessions"),
            error: None,
        },
        cams: Arc::new(Mutex::new(CamList::Loading)),
        waker: None,
        pipeline: None,
        live: Pixels::new(),
        mosaic: Pixels::new(),
        report: None,
        preview: None,
        paused: false,
        ended: None,
        details_open: false,
        export: None,
        export_msg: None,
        export_crop: false,
        export_scale: 0,
        export_format: 0,
        export_dialog: None,
        updater: update::Updater::new(),
        update_note: None,
        update_checking: false,
        initial: spec,
        fps_frames: 0,
        fps_t0: 0.0,
        fps: 0.0,
    };
    if let Some(kind) = fake {
        apply_fake(&mut app, &kind);
    }
    // Frozen start menu with the modes a real driver reported (visual testing only).
    if rune::extra_args().iter().any(|a| a == "--fake-menu") {
        let f = |fmts: &[(PixelFormat, u32)]| fmts.to_vec();
        let modes = vec![
            Mode { width: 1920, height: 1080, formats: f(&[(PixelFormat::Nv12, 60), (PixelFormat::Yuyv, 60)]) },
            Mode { width: 1280, height: 720, formats: f(&[(PixelFormat::Nv12, 60), (PixelFormat::Yuyv, 60)]) },
            Mode { width: 960, height: 540, formats: f(&[(PixelFormat::Nv12, 60), (PixelFormat::Yuyv, 60)]) },
        ];
        if let Ok(mut g) = app.cams.lock() {
            *g = CamList::Ready(vec![(0, "Elgato Virtual Camera".into()), (1, "USB CAMERA".into())]);
        }
        if let Ok(mut g) = app.menu.modes.lock() {
            *g = Modes::Ready(0, modes);
        }
        app.menu.cam = Some(0);
    }
    app
}

/// Starts the camera enumeration in the background and, if a CLI source was given, the pipeline.
fn setup(s: &mut App, waker: Waker) {
    s.waker = Some(waker.clone());
    s.updater.set_waker(waker.clone());
    if rune::extra_args().iter().any(|a| a == "--fake-menu" || a == "--fake") {
        return; // frozen states for visual testing: no camera, no network
    }
    update::cleanup();
    s.updater.check(false);
    let cams = s.cams.clone();
    std::thread::spawn(move || {
        #[cfg(feature = "camera")]
        let result = match mosaix::camera::nokhwa_source::list_cameras() {
            Ok(v) => CamList::Ready(v),
            Err(e) => CamList::Unavailable(format!("{e}")),
        };
        #[cfg(not(feature = "camera"))]
        let result = CamList::Unavailable("built without camera support".into());
        if let Ok(mut g) = cams.lock() {
            *g = result;
        }
        waker();
    });
    if let Some(spec) = s.initial.take() {
        start(s, spec);
    }
}

/// Reads the camera's real modes on a worker thread (the device is opened for the query).
fn request_modes(s: &mut App, index: u32) {
    let Some(waker) = s.waker.clone() else { return };
    if let Ok(mut g) = s.menu.modes.lock() {
        *g = Modes::Loading;
    }
    let modes = s.menu.modes.clone();
    std::thread::spawn(move || {
        #[cfg(feature = "camera")]
        let result = mosaix::camera::nokhwa_source::list_modes(index);
        #[cfg(not(feature = "camera"))]
        let result: anyhow::Result<Vec<Mode>> = Err(anyhow::anyhow!("built without camera support"));
        if let Ok(mut g) = modes.lock() {
            *g = match result {
                Ok(m) => Modes::Ready(index, m),
                Err(e) => Modes::Failed(index, format!("{e}")),
            };
        }
        waker();
    });
}

/// The modes to show for the selected camera, and a note (loading / could not read).
/// `None` while they are still being read.
fn visible_modes(s: &App) -> (Option<Vec<Mode>>, Option<String>) {
    let Some(cam) = s.menu.cam else { return (None, None) };
    match s.menu.modes.lock().map(|g| g.clone()).unwrap_or(Modes::Idle) {
        Modes::Ready(i, m) if i == cam => (Some(m), None),
        Modes::Failed(i, e) if i == cam => {
            (Some(fallback_modes()), Some(format!("Could not read the camera's modes ({e}); showing common ones.")))
        }
        _ => (None, Some("Detecting the camera's resolutions…".to_string())),
    }
}

/// Keeps the chosen resolution and format among what the camera really offers.
fn reconcile_modes(s: &mut App) {
    let Some(list) = visible_modes(s).0 else { return };
    if list.is_empty() {
        return;
    }
    let valid = s.menu.res.is_some_and(|(w, h)| list.iter().any(|m| m.width == w && m.height == h));
    if !valid {
        s.menu.res = list.iter().find(|m| (m.width, m.height) == (1920, 1080)).or(list.first()).map(|m| (m.width, m.height));
    }
    if let Some(mode) = list.iter().find(|m| Some((m.width, m.height)) == s.menu.res) {
        let ok = s.menu.format.is_some_and(|f| mode.formats.iter().any(|(pf, _)| *pf == f));
        if !ok {
            // Prefer an uncompressed format (no JPEG artefacts), MJPEG last of the usual ones.
            let rank = |f: PixelFormat| match f {
                PixelFormat::Yuyv => 0,
                PixelFormat::Nv12 => 1,
                PixelFormat::Mjpeg => 2,
                _ => 3,
            };
            s.menu.format = mode.formats.iter().map(|x| x.0).min_by_key(|f| rank(*f));
        }
    }
}

fn start(s: &mut App, spec: SourceSpec) {
    let Some(waker) = s.waker.clone() else { return };
    s.live = Pixels::new();
    s.mosaic = Pixels::new();
    s.report = None;
    s.preview = None;
    s.paused = false;
    s.ended = None;
    let root = Some(s.menu.save_dir.clone());
    s.pipeline = Some(Pipeline::start(spec, root, Params::default(), 1600, move || waker()));
    s.screen = Screen::Running;
}

fn back_to_menu(s: &mut App) {
    s.pipeline = None; // Drop stops the threads
    s.screen = Screen::Menu;
}

/// Drains messages from the analysis thread; `true` when the view must be rebuilt.
fn poll(s: &mut App, clock: f64, tx: &mut Tx) -> bool {
    let mut dirty = s.screen == Screen::Menu; // camera list may have changed
    if s.menu.cam.is_none() {
        let first = match s.cams.lock() {
            Ok(g) => match &*g {
                CamList::Ready(list) => list.first().map(|(i, _)| *i),
                _ => None,
            },
            Err(_) => None,
        };
        if let Some(i) = first {
            s.menu.cam = Some(i);
            request_modes(s, i);
        }
    }
    reconcile_modes(s);
    // Export thread: keep the progress moving, collect the result.
    let finished = s.export.as_ref().and_then(|j| j.result.lock().ok().and_then(|mut g| g.take()));
    if s.export.is_some() {
        dirty = true;
    }
    if let Some(r) = finished {
        s.export_msg = Some(match r {
            Ok(r) => (
                format!(
                    "Saved {} — {}×{} px, {}, {} overlaps aligned (max correction {:.1} px)",
                    r.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                    r.width,
                    r.height,
                    count(r.tiles, "tile"),
                    r.pairs_used,
                    r.max_correction
                ),
                true,
            ),
            Err(e) => (e, false),
        });
        s.export = None;
    }
    if s.updater.poll(tx) {
        dirty = true;
    }
    if let Some(note) = s.updater.notice.take() {
        s.update_note = Some(note);
    }
    if s.update_checking && !matches!(s.updater.state, update::State::Checking) {
        s.update_checking = false;
        dirty = true;
    }
    let Some(p) = &s.pipeline else { return dirty };
    while let Ok(ev) = p.events.try_recv() {
        dirty = true;
        match ev {
            Event::Update(u) => {
                let f = &*u.frame;
                let k = f.width().div_ceil(LIVE_MAX_W).max(1);
                let small;
                let img = if k > 1 {
                    small = mosaix::tracking::rgb_downscale(f, k);
                    &small
                } else {
                    f
                };
                s.live.set(img.width(), img.height(), rgb_to_rgba(img));
                s.report = Some(u.report);
                if let Some(pv) = u.preview {
                    s.mosaic.set(pv.image.width(), pv.image.height(), pv.image.as_raw().clone());
                    s.preview = Some(pv);
                }
                s.fps_frames += 1;
                if clock - s.fps_t0 >= 1.0 {
                    s.fps = s.fps_frames as f32 / (clock - s.fps_t0) as f32;
                    s.fps_frames = 0;
                    s.fps_t0 = clock;
                }
            }
            Event::Ended(msg) => s.ended = Some(msg),
        }
    }
    dirty
}

/// "1 tile", "3 tiles".
fn count(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

fn rgb_to_rgba(img: &image::RgbImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.as_raw().len() / 3 * 4);
    for p in img.as_raw().chunks_exact(3) {
        out.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    out
}

/// Starts an export of `session_dir` into `out` on a background thread.
fn start_export(s: &mut App, session_dir: PathBuf, out: PathBuf) {
    let Some(waker) = s.waker.clone() else { return };
    let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || waker());
    let progress = Progress::new(Some(wake.clone()));
    let result: Arc<Mutex<Option<Result<ExportResult, String>>>> = Arc::new(Mutex::new(None));
    let opts = ExportOptions { crop: s.export_crop, scale: EXPORT_SCALES[s.export_scale].0, ..Default::default() };
    s.export_msg = None;
    s.export = Some(ExportJob { progress: progress.clone(), result: result.clone() });
    std::thread::spawn(move || {
        let r = export_session(&session_dir, &out, &opts, &progress).map_err(|e| format!("{e:#}"));
        if let Ok(mut g) = result.lock() {
            *g = Some(r);
        }
        wake();
    });
}

/// Asks where to save, then exports `session_dir` there.
/// First free `mosaic.<ext>`, `mosaic-2.<ext>`… in the session folder.
fn unique_output(dir: &Path, ext: &str) -> PathBuf {
    let first = dir.join(format!("mosaic.{ext}"));
    if !first.exists() {
        return first;
    }
    (2..).map(|n| dir.join(format!("mosaic-{n}.{ext}"))).find(|p| !p.exists()).unwrap_or(first)
}

/// "Export" pressed in the options window: the mosaic goes into the session folder.
fn confirm_export(s: &mut App) {
    let Some(dir) = s.export_dialog.take() else { return };
    let ext = EXPORT_FORMATS[s.export_format].2;
    let out = unique_output(&dir, ext);
    start_export(s, dir, out);
}

fn cancel_dialog(s: &mut App) {
    s.export_dialog = None;
}
fn cancel_export(s: &mut App) {
    if let Some(j) = &s.export {
        j.progress.cancel();
    }
}

fn toggle_pause(s: &mut App) {
    s.paused = !s.paused;
    if let Some(p) = &s.pipeline {
        p.send(Control::Pause(s.paused));
    }
}

fn undo(s: &mut App) {
    if let Some(p) = &s.pipeline {
        p.send(Control::UndoLast);
    }
}

fn send(s: &App, c: Control) {
    if let Some(p) = &s.pipeline {
        p.send(c);
    }
}
fn end_row(s: &mut App) {
    send(s, Control::EndRow);
}
fn finish(s: &mut App) {
    send(s, Control::Finish);
}
fn reverse(s: &mut App) {
    send(s, Control::Reverse);
}
fn resume(s: &mut App) {
    send(s, Control::Resume);
}
fn on_key(s: &mut App, event: &KeyEvent, _: &mut Tx) -> bool {
    if s.screen != Screen::Running {
        return false;
    }
    match &event.key {
        Key::Text(t) if t == " " => toggle_pause(s),
        Key::Text(t) if t == "u" => undo(s),
        Key::Text(t) if t == "e" => end_row(s),
        Key::Text(t) if t == "f" => {
            if s.report.as_ref().is_some_and(|r| r.finished) {
                resume(s)
            } else {
                finish(s)
            }
        }
        Key::Text(t) if t == "r" => reverse(s),
        _ => return false,
    }
    true
}

// ---------------------------------------------------------------- widgets
fn button(key: String, label: String, on_press: impl Fn(&mut App) + 'static) -> El {
    el().key(key)
        .role("button")
        .label(label.clone())
        .focusable(true)
        .radius(8.0)
        .padding(8.0)
        .fill(hex("#24252b"))
        .on_press(on_press)
        .child(text(label).font(type_::BODY).color(Color::WHITE))
}

fn primary(key: &str, label: &str, on_press: impl Fn(&mut App) + 'static) -> El {
    el().key(key.to_string())
        .role("button")
        .label(label.to_string())
        .focusable(true)
        .radius(8.0)
        .padding(10.0)
        .fill(tok::accent(0, true))
        .on_press(on_press)
        .child(text(label.to_string()).font(type_::BODY).color(Color::WHITE))
}

fn choice(key: String, label: String, selected: bool, on_press: impl Fn(&mut App) + 'static) -> El {
    el().key(key)
        .role("button")
        .label(label.clone())
        .focusable(true)
        .radius(8.0)
        .padding(8.0)
        .fill(if selected { blue().alpha(0.35) } else { hex("#24252b") })
        .on_press(on_press)
        .child(text(label).font(type_::BODY).color(Color::WHITE))
}

fn section(title: &str) -> El {
    col().w_full().gap(8.0).child(text(title.to_string()).font(type_::TITLE).color(Color::WHITE))
}

fn gauge(label: String, frac: f32, color: Color, marker: Option<f32>) -> El {
    let frac = frac.clamp(0.0, 1.0);
    let mut bar = el()
        .w(GAUGE_W)
        .h(5.0)
        .radius(3.0)
        .fill(Color::WHITE.alpha(0.2))
        .child(el().absolute(0.0, 0.0).w((GAUGE_W * frac).max(1.0)).h(5.0).radius(3.0).fill(color));
    if let Some(m) = marker {
        bar = bar.child(el().absolute(GAUGE_W * m, -3.0).w(2.0).h(11.0).fill(Color::WHITE));
    }
    col().gap(4.0).child(text(label).font(type_::BODY).color(soft())).child(bar)
}

fn kv(label: &str, value: String) -> El {
    row()
        .w_full()
        .gap(8.0)
        .child(text(label.to_string()).font(type_::BODY).color(dim()))
        .child(spacer())
        .child(text(value).font(type_::BODY).color(soft()))
}

// ---------------------------------------------------------------- export widgets
fn export_status(s: &App) -> Option<El> {
    if let Some(j) = &s.export {
        Some(
            col()
                .w_full()
                .gap(6.0)
                .child(text(format!("{} — {:.0} %", j.progress.stage(), j.progress.fraction() * 100.0)).font(type_::BODY).color(amber()))
                .child(button("export-cancel".into(), "Cancel export".into(), cancel_export)),
        )
    } else {
        s.export_msg.as_ref().map(|(msg, ok)| text(msg.clone()).font(type_::BODY).color(if *ok { green() } else { red() }))
    }
}

/// Export block of the side panel: button + crop/scale options + progress or result.
fn export_panel(s: &App) -> El {
    let button = el()
        .key("export".to_string())
        .role("button")
        .label("Export mosaic".to_string())
        .focusable(true)
        .radius(8.0)
        .padding(10.0)
        .fill(tok::accent(0, true))
        .on_press(|s: &mut App| {
            if s.export.is_some() {
                return;
            }
            match s.pipeline.as_ref().and_then(|p| p.session_dir()) {
                Some(dir) => {
                    s.export_msg = None;
                    s.export_dialog = Some(dir);
                }
                None => s.export_msg = Some(("No session folder yet — capture a tile first".into(), false)),
            }
        })
        .child(text("Export…".to_string()).font(type_::BODY).color(Color::WHITE));
    let mut c = col().w_full().gap(8.0).child(button);
    if let Some(st) = export_status(s) {
        c = c.child(st);
    }
    c
}

/// A combo box: shows the current value; pressing it opens a native pick list.
fn combo(key: &str, value: String, on_press_tx: impl Fn(&mut App, &mut Tx) + 'static) -> El {
    el().key(key.to_string())
        .role("button")
        .label(value.clone())
        .focusable(true)
        .w_full()
        .radius(8.0)
        .padding(10.0)
        .fill(hex("#24252b"))
        .on_press_tx(on_press_tx)
        .child(
            row()
                .w_full()
                .gap(8.0)
                .child(text(value).font(type_::BODY).color(Color::WHITE))
                .child(spacer())
                .child(text("▾".to_string()).font(type_::BODY).color(dim())),
        )
}

/// Options window of the export (format, scale, crop). The images go into the session folder.
fn export_dialog_view(s: &App, dir: &Path) -> El {
    let mut formats = row().gap(8.0);
    for (i, (_, label, _)) in EXPORT_FORMATS.iter().enumerate() {
        formats =
            formats.child(choice(format!("fmt-{i}"), label.to_string(), s.export_format == i, move |s: &mut App| s.export_format = i));
    }
    let mut scales = row().gap(8.0);
    for (i, (_, label)) in EXPORT_SCALES.iter().enumerate() {
        scales = scales.child(choice(format!("scale-{i}"), label.to_string(), s.export_scale == i, move |s: &mut App| s.export_scale = i));
    }
    let crop = choice(
        "export-crop".into(),
        format!("Crop to the covered rectangle: {}", if s.export_crop { "on" } else { "off" }),
        s.export_crop,
        |s: &mut App| s.export_crop = !s.export_crop,
    );
    let card = col()
        .w(440.0)
        .gap(14.0)
        .padding(24.0)
        .radius(16.0)
        .fill(card())
        .child(text("Export mosaic".to_string()).font(type_::TITLE).color(Color::WHITE))
        .child(text(format!("The image is written to {}", dir.display())).font(type_::BODY).color(dim()))
        .child(text("Format".to_string()).font(type_::BODY).color(soft()))
        .child(formats)
        .child(text("Size".to_string()).font(type_::BODY).color(soft()))
        .child(scales)
        .child(crop)
        .child(row().gap(8.0).child(primary("export-go", "Export", confirm_export)).child(button(
            "export-cancel-dialog".into(),
            "Cancel".into(),
            cancel_dialog,
        )));
    el().key("export-modal".to_string())
        .absolute(0.0, 0.0)
        .w_full()
        .h_full()
        .fill(Color::BLACK.alpha(0.55))
        .center()
        .modal(true, cancel_dialog)
        .child(card)
}
// ---------------------------------------------------------------- menu
fn menu_view(s: &App) -> El {
    let t = tok::theme(true);

    // Camera section: camera combo, then the modes the camera really offers
    let mut cams = section("Camera");
    match s.cams.lock().map(|g| g.clone()).unwrap_or(CamList::Loading) {
        CamList::Loading => {
            cams = cams.child(text("Looking for cameras…".to_string()).font(type_::BODY).color(dim()));
        }
        CamList::Unavailable(e) => {
            cams = cams.child(text(format!("Cameras unavailable: {e}")).font(type_::BODY).color(dim()));
        }
        CamList::Ready(list) if list.is_empty() => {
            cams = cams.child(text("No camera found".to_string()).font(type_::BODY).color(dim()));
        }
        CamList::Ready(list) => {
            let name = list.iter().find(|(i, _)| Some(*i) == s.menu.cam).map_or("Select a camera".to_string(), |(_, n)| n.clone());
            let items: Vec<(String, String)> = list.iter().map(|(i, n)| (i.to_string(), n.clone())).collect();
            let selected = s.menu.cam.map(|i| i.to_string()).unwrap_or_default();
            cams = cams.child(combo("cam-combo", name, move |_: &mut App, tx: &mut Tx| {
                let menu = PopupMenu::picker(items.clone(), &selected);
                tx.popup_menu(menu, "cam-combo", |s: &mut App, picked, _| {
                    if let Some(i) = picked.and_then(|id| id.parse::<u32>().ok()) {
                        if s.menu.cam != Some(i) {
                            s.menu.cam = Some(i);
                            s.menu.res = None;
                            s.menu.format = None;
                            s.menu.error = None;
                            request_modes(s, i);
                        }
                    }
                });
            }));

            let (modes, note) = visible_modes(s);
            if let Some(n) = note {
                cams = cams.child(text(n).font(type_::BODY).color(dim()));
            }
            if let Some(modes) = modes {
                let title = |m: &Mode| format!("{} × {}  ·  up to {} fps", m.width, m.height, m.best_fps());
                let current =
                    modes.iter().find(|m| Some((m.width, m.height)) == s.menu.res).map_or("Select a resolution".to_string(), &title);
                let res_items: Vec<(String, String)> = modes.iter().map(|m| (format!("{}x{}", m.width, m.height), title(m))).collect();
                let res_selected = s.menu.res.map(|(w, h)| format!("{w}x{h}")).unwrap_or_default();
                cams = cams.child(combo("res-combo", current, move |_: &mut App, tx: &mut Tx| {
                    let menu = PopupMenu::picker(res_items.clone(), &res_selected);
                    tx.popup_menu(menu, "res-combo", |s: &mut App, picked, _| {
                        let parsed = picked.and_then(|id| {
                            let (w, h) = id.split_once('x')?;
                            Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
                        });
                        if let Some(r) = parsed {
                            s.menu.res = Some(r);
                            reconcile_modes(s);
                        }
                    });
                }));
                // Formats available at this resolution
                let mut fmt = row().gap(8.0);
                if let Some(mode) = modes.iter().find(|m| Some((m.width, m.height)) == s.menu.res) {
                    for &(pf, fps) in &mode.formats {
                        fmt = fmt.child(choice(
                            format!("fmt-{}", pf.label()),
                            format!("{} · {} fps", pf.label(), fps),
                            s.menu.format == Some(pf),
                            move |s: &mut App| s.menu.format = Some(pf),
                        ));
                    }
                }
                cams = cams.child(fmt).child(primary("start-cam", "Start camera", |s: &mut App| {
                    let Some(index) = s.menu.cam else {
                        s.menu.error = Some("Select a camera first".into());
                        return;
                    };
                    let (Some((width, height)), Some(format)) = (s.menu.res, s.menu.format) else {
                        s.menu.error = Some("Choose a resolution first".into());
                        return;
                    };
                    let fps = visible_modes(s)
                        .0
                        .and_then(|l| l.into_iter().find(|m| (m.width, m.height) == (width, height)))
                        .and_then(|m| m.formats.iter().find(|(pf, _)| *pf == format).map(|f| f.1))
                        .unwrap_or(30)
                        .clamp(1, 30);
                    start(s, SourceSpec::Camera { index, width, height, fps, format });
                }));
            }
        }
    }

    // Save folder: every captured tile and every exported image goes there
    let folder = section("Save folder")
        .child(text("Captured tiles and exported images are stored here.".to_string()).font(type_::BODY).color(dim()))
        .child(
            row()
                .w_full()
                .gap(8.0)
                .child(text(s.menu.save_dir.display().to_string()).font(type_::BODY).color(soft()))
                .child(spacer())
                .child(
                    el().key("pick-folder".to_string())
                        .role("button")
                        .label("Choose folder".to_string())
                        .focusable(true)
                        .radius(8.0)
                        .padding(8.0)
                        .fill(hex("#24252b"))
                        .on_press_tx(|_: &mut App, tx: &mut Tx| {
                            let options = FileDialogOptions { title: "Choose the save folder".into(), ..Default::default() };
                            tx.open_folder(options, |s: &mut App, result, _tx| match result {
                                Ok(Some(path)) => s.menu.save_dir = path,
                                Ok(None) => {}
                                Err(e) => s.menu.error = Some(e.to_string()),
                            });
                        })
                        .child(text("Choose…".to_string()).font(type_::BODY).color(Color::WHITE)),
                ),
        );

    let mut panel = col()
        .w(520.0)
        .gap(18.0)
        .padding(28.0)
        .radius(16.0)
        .fill(card())
        .child(text(APP_NAME.to_string()).font(type_::TITLE).color(Color::WHITE))
        .child(text("Choose a camera, then slide the object under it row by row.".to_string()).font(type_::BODY).color(soft()))
        .child(cams)
        .child(folder);
    if let Some(e) = &s.menu.error {
        panel = panel.child(text(e.clone()).font(type_::BODY).color(red()));
    }
    col().w_full().h_full().center().gap(12.0).fill(t.bg).child(panel).child(el().w(520.0).child(version_link()))
} // ---------------------------------------------------------------- running view
fn state_color(st: CaptureState) -> Color {
    match st {
        CaptureState::Tracking => soft(),
        CaptureState::Stabilizing => amber(),
        CaptureState::Capture => green(),
        CaptureState::Move => blue(),
        CaptureState::Lost => red(),
    }
}

/// The next thing to do: a short instruction and a line explaining it.
fn next_step(s: &App, r: &Report) -> (String, String) {
    let row = r.rows_done + 1;
    let along = if r.row_dir > 0.0 { "RIGHT" } else { "LEFT" };
    if s.paused {
        return ("Paused".into(), "Press Space to resume.".into());
    }
    if r.finished {
        return (format!("Map complete — {}", count(r.tiles, "tile")), "Export the mosaic, or press F to keep mapping.".into());
    }
    if r.state == CaptureState::Lost {
        return ("Tracking lost".into(), "Move back over an area already captured.".into());
    }
    match r.phase {
        Phase::Closing(Then::StepDown) => return ("Row end — hold still".into(), format!("Capturing the last tile of row {row}.")),
        Phase::Closing(Then::Finish) => return ("Map end — hold still".into(), "Capturing the last tile of the map.".into()),
        Phase::Down => {
            let next = if r.row_dir > 0.0 { "LEFT" } else { "RIGHT" };
            return (
                format!("Row {} done — move DOWN one step", r.rows_done),
                format!("Row {row} will run to the {next}. Last row done? Press F (Mapping finished)."),
            );
        }
        Phase::Row => {}
    }
    if r.tiles == 0 {
        return (
            "Hold still for the first tile".into(),
            "Start at the top-left corner of the area to map. Press E at the end of each row.".into(),
        );
    }
    match r.state {
        CaptureState::Stabilizing => ("Hold still…".into(), format!("Capturing tile {}.", r.tiles + 1)),
        CaptureState::Capture => (format!("Tile {} captured", r.tiles), format!("Row {row}: keep moving {along}.")),
        CaptureState::Tracking if r.sharp_rel < 0.8 => {
            ("Image is soft — adjust the focus".into(), "Capture waits for a sharp image.".into())
        }
        _ => (format!("Row {row}: move {along}"), "At the end of the row, press E (Row ends here).".into()),
    }
}

/// Colour of the instruction: guidance steps in amber, the map done in green, else the capture state.
fn step_color(s: &App, r: &Report) -> Color {
    if s.paused {
        dim()
    } else if r.finished {
        green()
    } else if r.state != CaptureState::Lost && r.phase != Phase::Row {
        amber()
    } else {
        state_color(r.state)
    }
}

/// Aim reticle: progress ring toward the capture threshold + direction arrow.
fn reticle(r: &Report, paused: bool) -> El {
    use std::f32::consts::{PI, TAU};
    let c = RETICLE / 2.0;
    let color = state_color(r.state);
    let progress = (r.offset_frac as f32 / 0.65).clamp(0.0, 1.0);

    // Annular sector as ONE filled polygon (outer arc, then inner arc reversed). Filling instead
    // of stroking a polyline avoids the beaded look that overlapping translucent round joints
    // give, and 4 segments per degree keeps the circle smooth at any display scale.
    let ring = |r_in: f32, r_out: f32, sweep: f32| -> Vec<[f32; 2]> {
        let n = ((sweep / TAU * 1440.0).ceil() as usize).max(2);
        let pt = |rad: f32, i: usize| {
            let a = -PI / 2.0 + sweep * i as f32 / n as f32;
            [c + rad * a.cos(), c + rad * a.sin()]
        };
        let mut v: Vec<[f32; 2]> = (0..=n).map(|i| pt(r_out, i)).collect();
        v.extend((0..=n).rev().map(|i| pt(r_in, i)));
        v
    };
    const FULL: f32 = TAU - 0.002; // tiny gap keeps the polygon simple

    // Arrow as a filled polygon pointing along `d`; `g` grows it (used for the dark outline).
    let arrow = |d: (f32, f32), g: f32| -> Vec<[f32; 2]> {
        let n = (-d.1, d.0);
        let (from, tip, head_len, shaft, head_w) = (64.0 - g, 118.0 + g, 26.0 + g, 4.0 + g, 15.0 + g);
        let at = |along: f32, side: f32| [c + d.0 * along + n.0 * side, c + d.1 * along + n.1 * side];
        let base = tip - head_len;
        vec![at(from, shaft), at(base, shaft), at(base, head_w), at(tip, 0.0), at(base, -head_w), at(base, -shaft), at(from, -shaft)]
    };

    let dark = Color::BLACK.alpha(0.45);
    let mut shape = Shape::new()
        // Track: dark halo + translucent white ring, so it reads on bright and dark surfaces.
        .fill(ring(40.0, 53.0, FULL), dark)
        .fill(ring(43.5, 49.5, FULL), Color::WHITE.alpha(0.35));
    if r.tiles > 0 && !paused {
        shape = shape.fill(ring(42.0, 51.0, (progress * TAU).clamp(0.02, FULL)), color);
    }
    // Crosshair with a gap at the centre, drawn twice (halo then white) for contrast.
    for (w, col) in [(3.0, dark), (1.0, Color::WHITE.alpha(0.85))] {
        shape = shape
            .line([c - 24.0, c], [c - 6.0, c], w, rune::LineCap::Butt, col)
            .line([c + 6.0, c], [c + 24.0, c], w, rune::LineCap::Butt, col)
            .line([c, c - 24.0], [c, c - 6.0], w, rune::LineCap::Butt, col)
            .line([c, c + 6.0], [c, c + 24.0], w, rune::LineCap::Butt, col);
    }
    if let (CaptureState::Move, Some(h), false) = (r.state, r.heading, paused) {
        // Arrow = where to go: the direction the view must move toward.
        let d = (h.0 as f32, h.1 as f32);
        shape = shape.fill(arrow(d, 3.0), dark).fill(arrow(d, 0.0), Color::WHITE);
    }
    canvas(shape).size(RETICLE, RETICLE)
}

fn live_overlay(s: &App) -> El {
    let Some(r) = &s.report else {
        return el();
    };
    let chip = row().gap(12.0).child(text(next_step(s, r).0).font(type_::BODY).color(step_color(s, r)));
    let hint = row().w_full().center().child(el().padding(10.0).radius(16.0).fill(chip_bg()).child(chip));
    let stab = (r.stable_secs / 0.25).clamp(0.0, 1.0) as f32;
    let sharp_color = if r.sharp_rel < 0.8 { red() } else { green() };
    let bar = row()
        .w_full()
        .gap(18.0)
        .padding(12.0)
        .radius(12.0)
        .fill(chip_bg())
        .child(gauge(format!("Sharpness {:.0} %", r.sharp_rel * 100.0), r.sharp_rel, sharp_color, Some(0.8)))
        .child(gauge(format!("Stability {:.0} ms", r.stable_secs * 1000.0), stab, amber(), None))
        .child(gauge(format!("Offset {:.0} %", r.offset_frac * 100.0), r.offset_frac as f32, blue(), Some(0.65)))
        .child(spacer())
        .child(text(count(r.tiles, "tile")).font(type_::TITLE).color(Color::WHITE));
    col().absolute(0.0, 0.0).w_full().h_full().padding(12.0).child(hint).child(spacer()).child(bar)
}

fn running_view(s: &App) -> El {
    let t = tok::theme(true);

    // ---- Left: live image + overlays
    let mut live = el()
        .key("live".to_string())
        .grow(1.0)
        .h_full()
        .radius(10.0)
        .clip_rounded(10.0)
        .fill(Color::BLACK)
        .pixels_fit(&s.live, SurfaceFit::Fit);
    if let Some(r) = &s.report {
        live = live.child(el().absolute(0.0, 0.0).w_full().h_full().center().child(reticle(r, s.paused)));
    }
    live = live.child(live_overlay(s));
    if let Some(msg) = &s.ended {
        live = live.child(el().absolute(0.0, 60.0).w_full().center().child(
            el().padding(10.0).radius(10.0).fill(chip_bg()).child(text(format!("Source stopped: {msg}")).font(type_::BODY).color(amber())),
        ));
    }

    // ---- Right: side panel
    let mut mosaic_box = el()
        .key("mosaic".to_string())
        .w(MOSAIC_BOX.0)
        .h(MOSAIC_BOX.1)
        .radius(10.0)
        .clip_rounded(10.0)
        .fill(Color::BLACK)
        .pixels_fit(&s.mosaic, SurfaceFit::Fit);
    if let (Some(pv), Some(r)) = (&s.preview, &s.report) {
        let (pw, ph) = (pv.image.width() as f32, pv.image.height() as f32);
        let f = (MOSAIC_BOX.0 / pw).min(MOSAIC_BOX.1 / ph);
        let (ox, oy) = ((MOSAIC_BOX.0 - pw * f) / 2.0, (MOSAIC_BOX.1 - ph * f) / 2.0);
        let sc = pv.scale as f32 * f;
        let x = ox + (r.pos.0 - pv.origin.0) as f32 * sc;
        let y = oy + (r.pos.1 - pv.origin.1) as f32 * sc;
        let (w, h) = (r.frame_size.0 as f32 * sc, r.frame_size.1 as f32 * sc);
        mosaic_box = mosaic_box.child(el().absolute(x, y).w(w).h(h).border(2.0, blue()));
    }

    let mut panel = col().w(PANEL_W).h_full().gap(12.0).child(mosaic_box);

    if let Some(r) = &s.report {
        // What to do next, in words
        let (title, detail) = next_step(s, r);
        let progress = if r.finished {
            let rows = r.rows_done.max(1);
            format!("{} · {}", count(rows, "row"), count(r.tiles, "tile"))
        } else {
            format!("Row {} · {}", r.rows_done + 1, count(r.tiles, "tile"))
        };
        panel = panel.child(
            col()
                .w_full()
                .gap(6.0)
                .padding(12.0)
                .radius(10.0)
                .fill(card())
                .child(text(progress).font(type_::BODY).color(dim()))
                .child(text(title).font(type_::TITLE).color(step_color(s, r)).w_full().wrap())
                .child(text(detail).font(type_::BODY).color(soft()).w_full().wrap()),
        );

        // Collapsible details: measurements and rarely needed settings
        let arrow = if s.details_open { "▾ Details" } else { "▸ Details" };
        let mut details = col().w_full().gap(6.0).child(
            el().key("details-toggle".to_string())
                .role("button")
                .label("Toggle details".to_string())
                .focusable(true)
                .on_press(|s: &mut App| s.details_open = !s.details_open)
                .child(text(arrow.to_string()).font(type_::BODY).color(dim())),
        );
        if s.details_open {
            let motion = r.motion.map_or("—".to_string(), |(dx, dy)| format!("{:.1} px", dx.hypot(dy)));
            details = details
                .child(button("reverse".into(), "Flip row direction (R)".into(), reverse))
                .child(kv("Motion", motion))
                .child(kv("Sharpness", format!("{:.0} %", r.sharp_rel * 100.0)))
                .child(kv("Stability", format!("{:.0} ms", r.stable_secs * 1000.0)))
                .child(kv("Offset from last tile", format!("{:.0} %", r.offset_frac * 100.0)))
                .child(kv("Position", format!("{:.0}, {:.0} px", r.pos.0, r.pos.1)))
                .child(kv("Analysis", format!("{:.1} ms", r.analysis_ms)))
                .child(kv("Display", format!("{:.0} fps", s.fps)));
        }
        panel = panel.child(details);
    } else {
        panel = panel.child(text("Opening source…".to_string()).font(type_::BODY).color(dim()));
    }

    // Row and map ends are given by hand: "row ends here" while on a row, "mapping finished"
    // at any time once something is captured.
    let mut guidance = col().w_full().gap(8.0);
    if let Some(r) = &s.report {
        if r.finished {
            guidance = guidance.child(button("resume".into(), "Continue mapping (F)".into(), resume));
        } else if r.tiles > 0 {
            if r.phase == Phase::Row {
                guidance = guidance.child(button("end-row".into(), "Row ends here (E)".into(), end_row));
            }
            if r.phase != Phase::Closing(Then::Finish) {
                guidance = guidance.child(primary("finish", "Mapping finished (F)", finish));
            }
        }
    }
    panel = panel.child(spacer()).child(guidance).child(export_panel(s)).child(
        col()
            .w_full()
            .gap(8.0)
            .child(
                row()
                    .gap(8.0)
                    .child(button("pause".into(), if s.paused { "Resume".into() } else { "Pause".into() }, toggle_pause))
                    .child(button("undo".into(), "Undo last tile".into(), undo)),
            )
            .child(button("menu".into(), "Change source".into(), back_to_menu))
            .child(version_link()),
    );

    row().w_full().h_full().gap(12.0).padding(12.0).fill(t.bg).child(live).child(panel)
}

// ---------------------------------------------------------------- updates
/// The version, pressed to look for a newer release.
fn version_link() -> El {
    row()
        .key("version".to_string())
        .role("button")
        .label("Check for updates".to_string())
        .focusable(true)
        .w_full()
        .center()
        .on_press_tx(|s: &mut App, tx: &mut Tx| {
            tx.instant();
            s.update_note = None;
            s.update_checking = true;
            s.updater.check(true);
        })
        .child(text(format!("Mosaïx {} · check for updates", update::VERSION)).font(type_::BODY).color(dim()))
}

/// A button of the update bar. Its changes land at once: the bar showing, hiding or changing
/// its text resizes the whole screen under it, which would otherwise spring and fade.
fn bar_button(key: &str, label: &str, primary: bool, on_press: impl Fn(&mut App) + 'static) -> El {
    el().key(key.to_string())
        .role("button")
        .label(label.to_string())
        .focusable(true)
        .radius(8.0)
        .padding(8.0)
        .fill(if primary { tok::accent(0, true) } else { hex("#24252b") })
        .on_press_tx(move |s: &mut App, tx: &mut Tx| {
            tx.instant();
            on_press(s);
        })
        .child(text(label.to_string()).font(type_::BODY).color(Color::WHITE))
}

/// A bar across the top when a newer release exists (or during and after a check asked for by
/// hand), with what can be done about it.
fn update_banner(s: &App) -> Option<El> {
    use update::State;
    let open_page = |s: &mut App| {
        if let Some(url) = s.updater.page() {
            update::open_url(&url);
        }
    };
    let close = || {
        bar_button("update-close", "Dismiss", false, |s: &mut App| {
            s.updater.dismiss();
            s.update_note = None;
        })
    };
    let (message, buttons): (String, Vec<El>) = match (&s.update_note, &s.updater.state) {
        (_, State::Checking) if s.update_checking => ("Checking for updates…".into(), Vec::new()),
        (Some(note), State::Idle | State::Checking) => (note.clone(), vec![close()]),
        (_, State::Available(o)) if o.can_install() => (
            format!("Mosaïx {} is available (you have {}).", o.release.version, update::VERSION),
            vec![
                bar_button("update-install", "Update now", true, |s: &mut App| s.updater.install()),
                bar_button("update-page", "Release page", false, open_page),
                close(),
            ],
        ),
        (_, State::Available(o)) => (
            format!("Mosaïx {} is available (you have {}). Download it from its release page.", o.release.version, update::VERSION),
            vec![bar_button("update-page", "Open release page", true, open_page), close()],
        ),
        (_, State::Downloading { offer, done, total }) => {
            let pct = if *total > 0 { format!(" {} %", (done * 100 / total).min(100)) } else { String::new() };
            (format!("Downloading Mosaïx {}…{pct}", offer.release.version), Vec::new())
        }
        (_, State::Installing(offer)) => (format!("Installing Mosaïx {}… the app restarts by itself.", offer.release.version), Vec::new()),
        (_, State::Failed { offer, message }) => (
            format!("Couldn't update to Mosaïx {}: {message}", offer.release.version),
            vec![bar_button("update-page", "Open release page", true, open_page), close()],
        ),
        (None, State::Idle | State::Checking) => return None,
    };
    Some(
        row()
            .key("update-banner".to_string())
            .w_full()
            .gap(8.0)
            .padding(8.0)
            .items_center()
            .radius(10.0)
            .fill(card())
            .child(el().grow(1.0).min_w(0.0).child(text(message).font(type_::BODY).color(Color::WHITE).truncate()))
            .children(buttons),
    )
}

fn view(s: &App, _: &mut Cx<App>) -> El {
    let main = match s.screen {
        Screen::Menu => menu_view(s),
        Screen::Running => running_view(s),
    };
    // The screen keeps the same place in the tree whether the update bar or the export window
    // shows: moving it to another parent would remount it, and its live view would fade back in.
    // The page carries the app's background, so the margin around the update bar isn't the
    // window's black.
    let mut page = col().key("page".to_string()).w_full().h_full().fill(tok::theme(true).bg);
    if let Some(bar) = update_banner(s) {
        page = page.child(el().key("update-slot".to_string()).w_full().padding(12.0).pb(0.0).child(bar));
    }
    page = page.child(el().key("screen".to_string()).w_full().grow(1.0).min_h(0.0).child(main));
    let mut root = el().key("root".to_string()).w_full().h_full().child(page);
    if let Some(dir) = &s.export_dialog {
        root = root.child(export_dialog_view(s, dir));
    }
    root
}
/// `--export <session_dir> <out> [--scale N] [--crop] [--no-refine]`: no window.
fn cli_export(args: &[String]) -> Option<i32> {
    let at = args.iter().position(|a| a == "--export")?;
    let (Some(dir), Some(out)) = (args.get(at + 1), args.get(at + 2)) else {
        eprintln!(
            "usage: mosaix --export <session_dir> <out.png|jpg|tif> [--scale 0.5] [--crop] [--no-refine] [--fine-side 1024] [--coarse-k 2]"
        );
        return Some(2);
    };
    let mut opts = ExportOptions::default();
    if let Some(i) = args.iter().position(|a| a == "--scale") {
        opts.scale = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    }
    if let Some(i) = args.iter().position(|a| a == "--fine-side") {
        opts.fine_side = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(opts.fine_side);
    }
    if let Some(i) = args.iter().position(|a| a == "--coarse-k") {
        opts.coarse_k = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(opts.coarse_k);
    }
    opts.crop = args.iter().any(|a| a == "--crop");
    opts.refine = !args.iter().any(|a| a == "--no-refine");
    let progress = Progress::new(None);
    let watcher = {
        let p = progress.clone();
        std::thread::spawn(move || {
            let mut last = String::new();
            while p.fraction() < 1.0 && !p.is_cancelled() {
                let line = format!("{} — {:.0} %", p.stage(), p.fraction() * 100.0);
                if line != last {
                    eprintln!("{line}");
                    last = line;
                }
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        })
    };
    let result = export_session(std::path::Path::new(dir), std::path::Path::new(out), &opts, &progress);
    progress.cancel(); // stops the progress printer
    let _ = watcher.join();
    match result {
        Ok(r) => {
            println!(
                "{}: {}x{} px (scale {:.2}), {} tiles, {} overlaps aligned, max correction {:.1} px, overlap error {}",
                r.path.display(),
                r.width,
                r.height,
                r.scale,
                r.tiles,
                r.pairs_used,
                r.max_correction,
                r.overlap_error.map_or("n/a".to_string(), |e| format!("{e:.2} levels"))
            );
            Some(0)
        }
        Err(e) => {
            eprintln!("export failed: {e:#}");
            Some(1)
        }
    }
}

/// `--list-modes`: prints every camera and the modes its driver really offers.
fn cli_list_modes(args: &[String]) -> Option<i32> {
    if !args.iter().any(|a| a == "--list-modes") {
        return None;
    }
    #[cfg(feature = "camera")]
    {
        match mosaix::camera::nokhwa_source::list_cameras() {
            Ok(cams) if cams.is_empty() => println!("no camera found"),
            Ok(cams) => {
                for (i, name) in cams {
                    println!("camera {i}: {name}");
                    match mosaix::camera::nokhwa_source::list_modes(i) {
                        Ok(modes) => {
                            for m in modes {
                                let f: Vec<String> = m.formats.iter().map(|(pf, fps)| format!("{} @{fps}", pf.label())).collect();
                                println!("  {} x {}  {}", m.width, m.height, f.join(", "));
                            }
                        }
                        Err(e) => println!("  modes unavailable: {e}"),
                    }
                }
            }
            Err(e) => println!("cannot list cameras: {e}"),
        }
        Some(0)
    }
    #[cfg(not(feature = "camera"))]
    {
        eprintln!("built without camera support");
        Some(1)
    }
}

/// `--grab <index> <WxH> <mjpeg|yuyv|nv12> [frames]`: opens the camera, grabs a few frames and
/// reports their size and the time per frame (checks that a mode really streams and decodes).
fn cli_grab(args: &[String]) -> Option<i32> {
    let at = args.iter().position(|a| a == "--grab")?;
    let (Some(idx), Some(res), Some(fmt)) = (args.get(at + 1), args.get(at + 2), args.get(at + 3)) else {
        eprintln!("usage: mosaix --grab <index> <WxH> <mjpeg|yuyv|nv12> [frames]");
        return Some(2);
    };
    let frames: usize = args.get(at + 4).and_then(|v| v.parse().ok()).unwrap_or(10);
    let (w, h) = res.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))).unwrap_or((1920, 1080));
    let format = match fmt.to_ascii_lowercase().as_str() {
        "yuyv" => PixelFormat::Yuyv,
        "nv12" => PixelFormat::Nv12,
        _ => PixelFormat::Mjpeg,
    };
    let spec = SourceSpec::Camera { index: idx.parse().unwrap_or(0), width: w, height: h, fps: 30, format };
    let mut src = match spec.open() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open the camera: {e:#}");
            return Some(1);
        }
    };
    println!("opened: {}", src.name());
    let t0 = std::time::Instant::now();
    for i in 0..frames {
        match src.next_frame() {
            Ok(Some(f)) => {
                if i == 0 || i + 1 == frames {
                    let mean = f.as_raw().iter().step_by(97).map(|&v| v as f64).sum::<f64>() / (f.as_raw().len() / 97) as f64;
                    println!("frame {}: {}x{}, mean level {:.1}", i + 1, f.width(), f.height(), mean);
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("frame {} failed: {e:#}", i + 1);
                return Some(1);
            }
        }
    }
    println!("{:.1} ms per frame", t0.elapsed().as_secs_f64() * 1000.0 / frames as f64);
    Some(0)
}

/// A windowed app has no console: the command-line modes write to the terminal they were
/// started from instead (nothing happens when there is none, e.g. a double-click).
fn attach_parent_console() {
    #[cfg(all(windows, not(debug_assertions)))]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn AttachConsole(process_id: u32) -> i32;
        }
        const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
        // SAFETY: plain Win32 call with a constant argument; failure only means no console.
        unsafe {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| matches!(a.as_str(), "--export" | "--list-modes" | "--grab")) {
        attach_parent_console();
    }
    if let Some(code) = cli_list_modes(&args) {
        std::process::exit(code);
    }
    if let Some(code) = cli_grab(&args) {
        std::process::exit(code);
    }
    if let Some(code) = cli_export(&args) {
        std::process::exit(code);
    }
    tracing_subscriber::fmt().init();
    // The executable carries the icon on Windows (build.rs). On macOS the Dock takes it from
    // the PNG, which also covers a run outside the app bundle; Linux reads the .desktop file.
    #[cfg(windows)]
    rune::set_app_icon_resource(1);
    #[cfg(target_os = "macos")]
    if let Err(e) = rune::set_app_icon(include_bytes!("../assets/icon/icon-macos.png").as_slice()) {
        eprintln!("mosaix: {e}");
    }
    let spec =
        AppSpec { setup: Some(setup), poll: Some(poll), on_key: Some(on_key), ..AppSpec::new(APP_NAME, (1280.0, 780.0), state, view) };
    // macOS: a standard titled window. Rune's default draws the content under a transparent
    // titlebar, so the traffic lights and the titlebar band would cover the top of the view.
    // Windows and Linux keep their native frame either way.
    rune::App::new(spec).chrome(rune::Chrome { full_size_content: false, ..rune::Chrome::DEFAULT }).run();
}
