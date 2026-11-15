//! PCB Mapper — Rune UI: start menu (source picker), live view with overlays,
//! side panel (mosaic, capture steps, metrics, details).
//!
//! CLI flags skip the menu: `--demo`, `--replay <dir> [--fps N]`,
//! `--camera <index> [--res WxH] [--yuyv]`. `--no-save` disables session saving.
//! UI testing without any source: `--fake stabilize|move|lost|suggest-row|suggest-finish|closing|finished`
//! shows a frozen state.

use pcb_mapper::analysis::Report;
use pcb_mapper::camera::SourceSpec;
use pcb_mapper::capture::{Params, State as CaptureState};
use pcb_mapper::guidance::{Phase, Suggestion, Then};
use pcb_mapper::mosaic::Preview;
use pcb_mapper::pipeline::{Control, Event, Pipeline};
use rune::tok::{self, type_};
use rune::{
    canvas, col, el, row, spacer, text, AppSpec, Color, Cx, El, FileDialogOptions, Key, KeyEvent, Pixels, Shape,
    SurfaceFit, Tx, Waker,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Max width of the image uploaded to the GPU for the live view.
const LIVE_MAX_W: u32 = 960;
const PANEL_W: f32 = 290.0;
const MOSAIC_BOX: (f32, f32) = (PANEL_W, 160.0);
const GAUGE_W: f32 = 130.0;
const RETICLE: f32 = 260.0;
const RESOLUTIONS: [(u32, u32); 3] = [(1920, 1080), (1280, 720), (640, 480)];

// ---------------------------------------------------------------- colours
fn hex(s: &str) -> Color {
    rune::color::hex(s).unwrap_or(Color::WHITE)
}
fn amber() -> Color { hex("#f2c14e") }
fn green() -> Color { hex("#5fd68a") }
fn blue() -> Color { hex("#4c9bff") }
fn red() -> Color { hex("#e24b4a") }
fn dim() -> Color { hex("#8d8f98") }
fn soft() -> Color { hex("#c9cbd2") }
fn card() -> Color { hex("#1c1d22") }
fn chip_bg() -> Color { Color::BLACK.alpha(0.72) }

// ---------------------------------------------------------------- state
#[derive(Clone)]
enum CamList {
    Loading,
    Ready(Vec<(u32, String)>),
    Unavailable(String),
}

struct MenuState {
    cam: Option<u32>,
    res: usize,
    mjpeg: bool,
    replay_dir: Option<PathBuf>,
    save: bool,
    error: Option<String>,
}

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
    auto_detect: bool,
    initial: Option<(SourceSpec, bool)>,
    fps_frames: u32,
    fps_t0: f64,
    fps: f32,
}

fn parse_args() -> (Option<SourceSpec>, bool) {
    let args = rune::extra_args();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let save = !args.iter().any(|a| a == "--no-save");
    let spec = if let Some(dir) = get("--replay") {
        let fps = get("--fps").and_then(|v| v.parse().ok()).unwrap_or(30.0);
        Some(SourceSpec::Replay { dir: PathBuf::from(dir), fps, looped: false })
    } else if let Some(idx) = get("--camera") {
        let (width, height) = get("--res")
            .and_then(|r| r.split_once('x').map(|(a, b)| (a.parse().ok(), b.parse().ok())))
            .and_then(|(a, b)| a.zip(b))
            .unwrap_or((1920, 1080));
        Some(SourceSpec::Camera {
            index: idx.parse().unwrap_or(0),
            width,
            height,
            fps: 30,
            mjpeg: !args.iter().any(|a| a == "--yuyv"),
        })
    } else if args.iter().any(|a| a == "--demo") {
        Some(SourceSpec::Synthetic)
    } else {
        None
    };
    (spec, save)
}

/// Frozen UI state for visual testing (`--fake <state>`): no threads, no source.
fn apply_fake(app: &mut App, kind: &str) {
    use pcb_mapper::mosaic::Mosaic;
    let board = pcb_mapper::camera::synthetic_board(1920, 1080);
    let view = image::imageops::crop_imm(&board, 320, 180, 1280, 720).to_image();
    app.live.set(view.width(), view.height(), rgb_to_rgba(&view));
    let mut mosaic = Mosaic::new();
    mosaic.add(&image::imageops::crop_imm(&board, 0, 0, 1280, 720).to_image(), (0.0, 0.0));
    mosaic.add(&image::imageops::crop_imm(&board, 640, 360, 1280, 720).to_image(), (870.0, 0.0));
    if let Some(pv) = mosaic.render(1600) {
        app.mosaic.set(pv.image.width(), pv.image.height(), pv.image.as_raw().clone());
        app.preview = Some(Arc::new(pv));
    }
    let (state, offset, stable, sharp_rel, phase, suggestion, finished) = match kind {
        "move" => (CaptureState::Move, 0.42, 0.0, 0.92, Phase::Row, None, false),
        "lost" => (CaptureState::Lost, 0.30, 0.0, 0.41, Phase::Row, None, false),
        "suggest-row" => (CaptureState::Move, 0.40, 0.0, 0.95, Phase::Row, Some(Suggestion::EndRow), false),
        "suggest-finish" => (CaptureState::Move, 0.40, 0.0, 0.95, Phase::Down, Some(Suggestion::Finish), false),
        "closing" => (CaptureState::Stabilizing, 0.22, 0.12, 0.96, Phase::Closing(Then::StepDown), None, false),
        "finished" => (CaptureState::Tracking, 0.15, 0.0, 0.96, Phase::Row, None, true),
        _ => (CaptureState::Stabilizing, 0.68, 0.18, 0.96, Phase::Row, None, false),
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
        phase,
        suggestion,
        rows_done: 1,
        finished,
        auto_detect: true,
    });
    app.screen = Screen::Running;
}

fn state() -> App {
    let (spec, save) = parse_args();
    let fake = rune::extra_args().iter().position(|a| a == "--fake").and_then(|i| rune::extra_args().get(i + 1).cloned());
    let mut app = App {
        screen: Screen::Menu,
        menu: MenuState { cam: None, res: 0, mjpeg: true, replay_dir: None, save, error: None },
        cams: Arc::new(Mutex::new(CamList::Loading)),
        waker: None,
        pipeline: None,
        live: Pixels::new(),
        mosaic: Pixels::new(),
        report: None,
        preview: None,
        paused: false,
        ended: None,
        details_open: true,
        auto_detect: true,
        initial: spec.map(|s| (s, save)),
        fps_frames: 0,
        fps_t0: 0.0,
        fps: 0.0,
    };
    if let Some(kind) = fake {
        apply_fake(&mut app, &kind);
    }
    app
}

/// Starts the camera enumeration in the background and, if a CLI source was given, the pipeline.
fn setup(s: &mut App, waker: Waker) {
    s.waker = Some(waker.clone());
    let cams = s.cams.clone();
    std::thread::spawn(move || {
        #[cfg(feature = "camera")]
        let result = match pcb_mapper::camera::nokhwa_source::list_cameras() {
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
    if let Some((spec, save)) = s.initial.take() {
        s.menu.save = save;
        start(s, spec);
    }
}

fn start(s: &mut App, spec: SourceSpec) {
    let Some(waker) = s.waker.clone() else { return };
    s.live = Pixels::new();
    s.mosaic = Pixels::new();
    s.report = None;
    s.preview = None;
    s.paused = false;
    s.auto_detect = true;
    s.ended = None;
    let root = s.menu.save.then(|| PathBuf::from("sessions"));
    s.pipeline = Some(Pipeline::start(spec, root, Params::default(), 1600, move || waker()));
    s.screen = Screen::Running;
}

fn back_to_menu(s: &mut App) {
    s.pipeline = None; // Drop stops the threads
    s.screen = Screen::Menu;
}

/// Drains messages from the analysis thread; `true` when the view must be rebuilt.
fn poll(s: &mut App, clock: f64, _: &mut Tx) -> bool {
    let mut dirty = s.screen == Screen::Menu; // camera list may have changed
    let Some(p) = &s.pipeline else { return dirty };
    while let Ok(ev) = p.events.try_recv() {
        dirty = true;
        match ev {
            Event::Update(u) => {
                let f = &*u.frame;
                let k = f.width().div_ceil(LIVE_MAX_W).max(1);
                let small;
                let img = if k > 1 {
                    small = pcb_mapper::tracking::rgb_downscale(f, k);
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

fn rgb_to_rgba(img: &image::RgbImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.as_raw().len() / 3 * 4);
    for p in img.as_raw().chunks_exact(3) {
        out.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    out
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
fn confirm(s: &mut App) {
    send(s, Control::Confirm);
}
fn dismiss(s: &mut App) {
    send(s, Control::Dismiss);
}
fn resume(s: &mut App) {
    send(s, Control::Resume);
}
fn toggle_auto(s: &mut App) {
    s.auto_detect = !s.auto_detect;
    send(s, Control::AutoDetect(s.auto_detect));
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
        Key::Text(t) if t == "a" => toggle_auto(s),
        Key::Enter => confirm(s),
        Key::Escape => dismiss(s),
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
    let mut bar = el().w(GAUGE_W).h(5.0).radius(3.0).fill(Color::WHITE.alpha(0.2)).child(
        el().absolute(0.0, 0.0).w((GAUGE_W * frac).max(1.0)).h(5.0).radius(3.0).fill(color),
    );
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

fn tile(label: &str, value: String) -> El {
    col()
        .grow(1.0)
        .gap(2.0)
        .padding(8.0)
        .radius(8.0)
        .fill(card())
        .child(text(label.to_string()).font(type_::BODY).color(dim()))
        .child(text(value).font(type_::TITLE).color(Color::WHITE))
}

// ---------------------------------------------------------------- menu
fn menu_view(s: &App) -> El {
    let t = tok::theme(true);

    // Camera section
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
            for (idx, name) in list {
                cams = cams.child(choice(format!("cam-{idx}"), name, s.menu.cam == Some(idx), move |s: &mut App| {
                    s.menu.cam = Some(idx);
                    s.menu.error = None;
                }));
            }
            let mut res = row().gap(8.0);
            for (i, (w, h)) in RESOLUTIONS.iter().enumerate() {
                res = res.child(choice(format!("res-{i}"), format!("{w}×{h}"), s.menu.res == i, move |s: &mut App| {
                    s.menu.res = i
                }));
            }
            let fmt = row()
                .gap(8.0)
                .child(choice("fmt-mjpeg".into(), "MJPEG".into(), s.menu.mjpeg, |s: &mut App| s.menu.mjpeg = true))
                .child(choice("fmt-yuyv".into(), "YUYV".into(), !s.menu.mjpeg, |s: &mut App| s.menu.mjpeg = false));
            cams = cams.child(res).child(fmt).child(primary("start-cam", "Start camera", |s: &mut App| {
                let Some(index) = s.menu.cam else {
                    s.menu.error = Some("Select a camera first".into());
                    return;
                };
                let (width, height) = RESOLUTIONS[s.menu.res];
                let spec = SourceSpec::Camera { index, width, height, fps: 30, mjpeg: s.menu.mjpeg };
                start(s, spec);
            }));
        }
    }

    // Replay section
    let folder = s.menu.replay_dir.as_ref().map(|p| p.display().to_string());
    let replay = section("Replay")
        .child(
            text(folder.unwrap_or_else(|| "No folder selected".into()))
                .font(type_::BODY)
                .color(dim()),
        )
        .child(
            row()
                .gap(8.0)
                .child(
                    el().key("pick-folder".to_string())
                        .role("button")
                        .label("Choose folder".to_string())
                        .focusable(true)
                        .radius(8.0)
                        .padding(8.0)
                        .fill(hex("#24252b"))
                        .on_press_tx(|_: &mut App, tx: &mut Tx| {
                            let options = FileDialogOptions { title: "Choose an image folder".into(), ..Default::default() };
                            tx.open_folder(options, |s: &mut App, result, _tx| match result {
                                Ok(Some(path)) => {
                                    s.menu.replay_dir = Some(path);
                                    s.menu.error = None;
                                }
                                Ok(None) => {}
                                Err(e) => s.menu.error = Some(e.to_string()),
                            });
                        })
                        .child(text("Choose folder…".to_string()).font(type_::BODY).color(Color::WHITE)),
                )
                .child(primary("start-replay", "Start replay", |s: &mut App| {
                    match s.menu.replay_dir.clone() {
                        Some(dir) => start(s, SourceSpec::Replay { dir, fps: 30.0, looped: false }),
                        None => s.menu.error = Some("Choose a folder of images first".into()),
                    }
                })),
        );

    // Demo section
    let demo = section("Demo")
        .child(
            text("Synthetic board scanned in a serpentine pattern, no hardware needed.".to_string())
                .font(type_::BODY)
                .color(dim()),
        )
        .child(primary("start-demo", "Start demo", |s: &mut App| start(s, SourceSpec::Synthetic)));

    let save = choice(
        "save-toggle".into(),
        if s.menu.save { "Saving tiles to disk: on".into() } else { "Saving tiles to disk: off".into() },
        s.menu.save,
        |s: &mut App| s.menu.save = !s.menu.save,
    );

    let mut panel = col()
        .w(520.0)
        .gap(18.0)
        .padding(28.0)
        .radius(16.0)
        .fill(card())
        .child(text("PCB Mapper".to_string()).font(type_::TITLE).color(Color::WHITE))
        .child(text("Choose a video source to start mapping.".to_string()).font(type_::BODY).color(soft()))
        .child(cams)
        .child(replay)
        .child(demo)
        .child(save);
    if let Some(e) = &s.menu.error {
        panel = panel.child(text(e.clone()).font(type_::BODY).color(red()));
    }
    col().w_full().h_full().center().fill(t.bg).child(panel)
}

// ---------------------------------------------------------------- running view
fn state_color(st: CaptureState) -> Color {
    match st {
        CaptureState::Tracking => soft(),
        CaptureState::Stabilizing => amber(),
        CaptureState::Capture => green(),
        CaptureState::Move => blue(),
        CaptureState::Lost => red(),
    }
}

/// Name of the direction the view must move toward (the camera heading).
fn heading_name(h: (f64, f64)) -> &'static str {
    if h.0 > 0.5 {
        "right"
    } else if h.0 < -0.5 {
        "left"
    } else if h.1 > 0.5 {
        "down"
    } else {
        "up"
    }
}

fn hint_text(s: &App, r: &Report) -> String {
    if s.paused {
        return "Paused".into();
    }
    if r.finished {
        return format!("Mapping complete — {} tiles", r.tiles);
    }
    match r.suggestion {
        Some(Suggestion::EndRow) => return "End of row detected?".into(),
        Some(Suggestion::Finish) => return "Bottom edge detected — finish the map?".into(),
        None => {}
    }
    if matches!(r.phase, Phase::Closing(_)) {
        return "Hold still — capturing the end".into();
    }
    match r.state {
        CaptureState::Tracking if r.sharp_rel < 0.8 => "Image is soft — adjust focus".into(),
        CaptureState::Tracking => "Tracking".into(),
        CaptureState::Stabilizing => "Hold still…".into(),
        CaptureState::Capture => "Captured".into(),
        CaptureState::Move => match r.heading {
            Some(h) => format!("Move {}", heading_name(h)),
            None => "Move".into(),
        },
        CaptureState::Lost => "Tracking lost — go back to an area already captured".into(),
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
        vec![
            at(from, shaft),
            at(base, shaft),
            at(base, head_w),
            at(tip, 0.0),
            at(base, -head_w),
            at(base, -shaft),
            at(from, -shaft),
        ]
    };

    let dark = Color::BLACK.alpha(0.45);
    let mut shape = Shape::new()
        // Track: dark halo + translucent white ring, so it reads on bright and dark boards.
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
    let color = if s.paused { dim() } else { state_color(r.state) };
    let hint = if r.suggestion.is_some() && !s.paused {
        // Automatic detection waiting for the user: confirm or dismiss.
        row().w_full().center().child(
            el().padding(10.0).radius(16.0).fill(chip_bg()).child(
                row()
                    .gap(12.0)
                    .child(text(hint_text(s, r)).font(type_::BODY).color(amber()))
                    .child(primary("confirm", "Confirm (Enter)", confirm))
                    .child(button("dismiss".into(), "Dismiss (Esc)".into(), dismiss)),
            ),
        )
    } else {
        row().w_full().center().child(
            el().padding(8.0).radius(16.0).fill(chip_bg()).child(text(hint_text(s, r)).font(type_::BODY).color(color)),
        )
    };
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
        .child(text(format!("{} tiles", r.tiles)).font(type_::TITLE).color(Color::WHITE));
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
        live = live.child(
            el().absolute(0.0, 60.0).w_full().center().child(
                el().padding(10.0).radius(10.0).fill(chip_bg()).child(
                    text(format!("Source stopped: {msg}")).font(type_::BODY).color(amber()),
                ),
            ),
        );
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
        // Capture steps
        let active = match r.state {
            CaptureState::Tracking => Some(0),
            CaptureState::Stabilizing => Some(1),
            CaptureState::Capture => Some(2),
            CaptureState::Move => Some(3),
            CaptureState::Lost => None,
        };
        let color = if s.paused { dim() } else { state_color(r.state) };
        let mut steps = row().w_full().gap(4.0).center();
        for i in 0..4usize {
            let on = active == Some(i);
            let done = active.is_some_and(|a| i < a);
            let fill = if on { color } else if done { green().alpha(0.3) } else { hex("#24252b") };
            let fg = if on { Color::BLACK } else if done { green() } else { dim() };
            steps = steps.child(
                el().w(22.0).h(22.0).radius(11.0).fill(fill).center().child(
                    text(if done { "✓".to_string() } else { (i + 1).to_string() }).font(type_::BODY).color(fg),
                ),
            );
            if i < 3 {
                steps = steps.child(el().grow(1.0).h(3.0).radius(2.0).fill(hex("#2a2b31")));
            }
        }
        let names = row()
            .w_full()
            .child(text("Track".to_string()).font(type_::BODY).color(dim()))
            .child(spacer())
            .child(text("Settle".to_string()).font(type_::BODY).color(dim()))
            .child(spacer())
            .child(text("Capture".to_string()).font(type_::BODY).color(dim()))
            .child(spacer())
            .child(text("Move".to_string()).font(type_::BODY).color(dim()));
        panel = panel.child(steps).child(names).child(text(hint_text(s, r)).font(type_::BODY).color(color));

        // Metric tiles
        let motion = r.motion.map_or("—".to_string(), |(dx, dy)| format!("{:.1} px", dx.hypot(dy)));
        panel = panel
            .child(
                row()
                    .w_full()
                    .gap(8.0)
                    .child(tile("Motion", motion))
                    .child(tile("Sharpness", format!("{:.0} %", r.sharp_rel * 100.0))),
            )
            .child(
                row()
                    .w_full()
                    .gap(8.0)
                    .child(tile("Stability", format!("{:.0} ms", r.stable_secs * 1000.0)))
                    .child(tile("Tiles", r.tiles.to_string())),
            );

        // Collapsible details
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
            details = details
                .child(kv("Row", format!("{}", r.rows_done + 1)))
                .child(kv("Position", format!("{:.0}, {:.0} px", r.pos.0, r.pos.1)))
                .child(kv("Offset from last tile", format!("{:.0} %", r.offset_frac * 100.0)))
                .child(kv("Raw sharpness", format!("{:.0}", r.sharp)))
                .child(kv("Analysis", format!("{:.1} ms", r.analysis_ms)))
                .child(kv("Display", format!("{:.0} fps", s.fps)));
        }
        panel = panel.child(details);
    } else {
        panel = panel.child(text("Opening source…".to_string()).font(type_::BODY).color(dim()));
    }

    let finished = s.report.as_ref().is_some_and(|r| r.finished);
    let stop_btn = if finished {
        primary("resume", "Resume (F)", resume)
    } else {
        button("finish".into(), "Finish (F)".into(), finish)
    };
    let guidance = col()
        .w_full()
        .gap(8.0)
        .child(text("Guidance".to_string()).font(type_::BODY).color(dim()))
        .child(
            row()
                .w_full()
                .gap(8.0)
                .child(button("end-row".into(), "End row (E)".into(), end_row))
                .child(button("reverse".into(), "Reverse (R)".into(), reverse))
                .child(stop_btn),
        )
        .child(choice(
            "auto".into(),
            format!("Auto-detect edges: {} (A)", if s.auto_detect { "on" } else { "off" }),
            s.auto_detect,
            toggle_auto,
        ));
    panel = panel.child(spacer()).child(guidance).child(
        col()
            .w_full()
            .gap(8.0)
            .child(
                row()
                    .gap(8.0)
                    .child(button(
                        "pause".into(),
                        if s.paused { "Resume".into() } else { "Pause".into() },
                        toggle_pause,
                    ))
                    .child(button("undo".into(), "Undo last tile".into(), undo)),
            )
            .child(button("menu".into(), "Change source".into(), back_to_menu)),
    );

    row().w_full().h_full().gap(12.0).padding(12.0).fill(t.bg).child(live).child(panel)
}

fn view(s: &App, _: &mut Cx<App>) -> El {
    match s.screen {
        Screen::Menu => menu_view(s),
        Screen::Running => running_view(s),
    }
}

fn main() {
    tracing_subscriber::fmt().init();
    rune::run(AppSpec {
        setup: Some(setup),
        poll: Some(poll),
        on_key: Some(on_key),
        ..AppSpec::new("PCB Mapper", (1280.0, 780.0), state, view)
    });
}
