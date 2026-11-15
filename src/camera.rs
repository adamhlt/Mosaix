//! Image sources: `CameraSource` abstraction + implementations
//! (image folder replay, synthetic test bench, nokhwa camera).

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Image source. No `Send` bound: some camera APIs are not
/// transferable between threads, so the source is created inside the capture thread
/// from a [`SourceSpec`].
pub trait CameraSource {
    fn name(&self) -> String;
    /// Next frame. `Ok(None)` = end of stream (replay finished).
    fn next_frame(&mut self) -> Result<Option<RgbImage>>;
}

/// Transferable description of a source, opened inside the capture thread.
#[derive(Debug, Clone)]
pub enum SourceSpec {
    /// Folder of images sorted by name, replayed at `fps`.
    Replay { dir: PathBuf, fps: f32, looped: bool },
    /// Synthetic board scanned in a serpentine pattern (demo without hardware).
    Synthetic,
    /// Real camera (`camera` feature).
    Camera { index: u32, width: u32, height: u32, fps: u32, mjpeg: bool },
}

impl SourceSpec {
    pub fn open(&self) -> Result<Box<dyn CameraSource>> {
        match self {
            SourceSpec::Replay { dir, fps, looped } => {
                Ok(Box::new(ReplaySource::from_dir(dir, *fps, *looped)?))
            }
            SourceSpec::Synthetic => Ok(Box::new(ViewportSource::demo())),
            #[cfg(feature = "camera")]
            SourceSpec::Camera { index, width, height, fps, mjpeg } => {
                Ok(Box::new(nokhwa_source::NokhwaSource::open(*index, *width, *height, *fps, *mjpeg)?))
            }
            #[cfg(not(feature = "camera"))]
            SourceSpec::Camera { .. } => {
                bail!("built without the `camera` feature (cargo build --features camera)")
            }
        }
    }
}

/// Throttles a read loop to a fixed rate.
struct Pacer {
    period: Duration,
    next: Instant,
}
impl Pacer {
    fn new(fps: f32) -> Self {
        Self { period: Duration::from_secs_f32(1.0 / fps.max(0.1)), next: Instant::now() }
    }
    fn wait(&mut self) {
        let now = Instant::now();
        if self.next > now {
            std::thread::sleep(self.next - now);
        }
        self.next = self.next.max(now) + self.period;
    }
}

/// Replays a folder of images.
pub struct ReplaySource {
    files: Vec<PathBuf>,
    idx: usize,
    looped: bool,
    pacer: Pacer,
    label: String,
}

impl ReplaySource {
    pub fn from_dir(dir: &Path, fps: f32, looped: bool) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading folder {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no image (png/jpg/tiff) in {}", dir.display());
        }
        Ok(Self { files, idx: 0, looped, pacer: Pacer::new(fps), label: format!("replay {}", dir.display()) })
    }
}

impl CameraSource for ReplaySource {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn next_frame(&mut self) -> Result<Option<RgbImage>> {
        if self.idx >= self.files.len() {
            if !self.looped {
                return Ok(None);
            }
            self.idx = 0;
        }
        self.pacer.wait();
        let p = &self.files[self.idx];
        self.idx += 1;
        Ok(Some(image::open(p).with_context(|| format!("image {}", p.display()))?.to_rgb8()))
    }
}

/// Waypoints of a serpentine sweep (top-left corner of the view, in board pixels).
/// `step` = fraction of the field covered between two captures (0.68 → overlap ≈ 32 %).
pub fn serpentine_path(board: (u32, u32), view: (u32, u32), step: f64) -> Vec<(f64, f64)> {
    let max_x = (board.0 - view.0) as f64;
    let max_y = (board.1 - view.1) as f64;
    let (sx, sy) = (view.0 as f64 * step, view.1 as f64 * step);
    let cols = (max_x / sx).ceil() as usize + 1;
    let rows = (max_y / sy).ceil() as usize + 1;
    let mut pts = Vec::new();
    for r in 0..rows {
        let y = (r as f64 * sy).min(max_y);
        let range: Vec<usize> = if r % 2 == 0 { (0..cols).collect() } else { (0..cols).rev().collect() };
        for c in range {
            pts.push(((c as f64 * sx).min(max_x), y));
        }
    }
    pts
}

/// Synthetic PCB image, deterministic, textured enough for phase correlation.
pub fn synthetic_board(w: u32, h: u32) -> RgbImage {
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut img = RgbImage::from_fn(w, h, |x, y| {
        // Slightly textured green background
        let n = ((x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663)) % 17) as i32 - 8;
        Rgb([(25 + n).clamp(0, 255) as u8, (95 + n).clamp(0, 255) as u8, (55 + n).clamp(0, 255) as u8])
    });
    let fill = |img: &mut RgbImage, x0: i64, y0: i64, rw: i64, rh: i64, c: [u8; 3]| {
        for y in y0.max(0)..(y0 + rh).min(h as i64) {
            for x in x0.max(0)..(x0 + rw).min(w as i64) {
                img.put_pixel(x as u32, y as u32, Rgb(c));
            }
        }
    };
    // Traces
    for _ in 0..(w * h / 12_000) {
        let (x, y) = ((rnd() * w as f64) as i64, (rnd() * h as f64) as i64);
        let len = (20.0 + rnd() * 160.0) as i64;
        let th = 3 + (rnd() * 5.0) as i64;
        let c = [190 + (rnd() * 30.0) as u8, 150 + (rnd() * 30.0) as u8, 40];
        if rnd() < 0.5 { fill(&mut img, x, y, len, th, c) } else { fill(&mut img, x, y, th, len, c) }
    }
    // Pads
    for _ in 0..(w * h / 6_000) {
        let (x, y) = ((rnd() * w as f64) as i64, (rnd() * h as f64) as i64);
        let s = 8 + (rnd() * 18.0) as i64;
        fill(&mut img, x, y, s, s, [215, 215, 200]);
        fill(&mut img, x + s / 3, y + s / 3, s / 3 + 1, s / 3 + 1, [30, 60, 40]);
    }
    // Components (black packages with rows of pins)
    for _ in 0..(w * h / 90_000) {
        let (x, y) = ((rnd() * w as f64) as i64, (rnd() * h as f64) as i64);
        let (cw, ch) = (40 + (rnd() * 120.0) as i64, 25 + (rnd() * 60.0) as i64);
        fill(&mut img, x, y, cw, ch, [20, 20, 24]);
        let mut px = x + 4;
        while px < x + cw - 6 {
            fill(&mut img, px, y - 6, 5, 6, [200, 200, 200]);
            fill(&mut img, px, y + ch, 5, 6, [200, 200, 200]);
            px += 10;
        }
        fill(&mut img, x + 6, y + 6, 8, 8, [90, 90, 100]);
    }
    img
}

/// Simulates a view moving over a large image: pauses at waypoints
/// (like a hand stopping for the capture), constant-speed movement in between.
pub struct ViewportSource {
    board: RgbImage,
    view: (u32, u32),
    path: Vec<(f64, f64)>,
    /// Pause (frames) at each waypoint.
    pauses: Vec<u32>,
    target: usize,
    pos: (f64, f64),
    speed: f64,
    pause_left: u32,
    pacer: Option<Pacer>,
    label: String,
}

impl ViewportSource {
    pub fn new(board: RgbImage, view: (u32, u32), step: f64, speed: f64, pause_frames: u32, fps: Option<f32>) -> Self {
        let path = serpentine_path((board.width(), board.height()), view, step);
        let pauses = vec![pause_frames; path.len()];
        Self::with_path(board, view, path, pauses, speed, fps)
    }

    /// Explicit waypoints (top-left of the view, canvas pixels) with a pause at each.
    pub fn with_path(
        board: RgbImage,
        view: (u32, u32),
        path: Vec<(f64, f64)>,
        pauses: Vec<u32>,
        speed: f64,
        fps: Option<f32>,
    ) -> Self {
        assert_eq!(path.len(), pauses.len());
        let pos = path[0];
        let pause_left = pauses[0];
        Self {
            board,
            view,
            path,
            pauses,
            target: 1,
            pos,
            speed,
            pause_left,
            pacer: fps.map(Pacer::new),
            label: "synthetic demo".into(),
        }
    }

    /// Demo: a PCB lying on a flat table, scanned in a serpentine pattern. At the end of each
    /// row the view overshoots the board edge and waits, so the edge detection has something to
    /// detect; the last step goes below the board to end the map.
    pub fn demo() -> Self {
        let (canvas, path, pauses) = demo_scene();
        Self::with_path(canvas, (1280, 720), path, pauses, 14.0, Some(30.0))
    }

    pub fn is_done(&self) -> bool {
        self.target >= self.path.len() && self.pause_left == 0
    }
}

/// Board on a table + waypoints of the demo hand. Pauses are short at ordinary waypoints and
/// long where the view overshoots an edge (time to detect, confirm and capture the last tile).
pub fn demo_scene() -> (RgbImage, Vec<(f64, f64)>, Vec<u32>) {
    const VIEW: (f64, f64) = (1280.0, 720.0);
    let (bw, bh) = (2700u32, 1700u32);
    let (mx, my) = (560u32, 320u32); // table margin around the board
    let mut canvas = RgbImage::from_fn(bw + 2 * mx, bh + 2 * my, |x, y| {
        let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) % 3) as i32 - 1;
        Rgb([(58 + n) as u8, (60 + n) as u8, (64 + n) as u8]) // dark, flat table
    });
    image::imageops::replace(&mut canvas, &synthetic_board(bw, bh), mx as i64, my as i64);

    let (mx, my, bw, bh) = (mx as f64, my as f64, bw as f64, bh as f64);
    let (sx, sy) = (0.68 * VIEW.0, 0.68 * VIEW.1);
    let x_right = mx + bw - 0.6 * VIEW.0; // 40 % of the view beyond the right edge
    let x_left = mx - 0.4 * VIEW.0; // 40 % beyond the left edge
    let y_bottom = my + bh - 0.6 * VIEW.1; // 40 % beyond the bottom edge
    let (short, edge) = (15u32, 150u32);

    let mut path = Vec::new();
    let mut pauses = Vec::new();
    let mut push = |p: (f64, f64), pause: u32| {
        path.push(p);
        pauses.push(pause);
    };
    for row in 0..3 {
        let y = my + row as f64 * sy;
        if row % 2 == 0 {
            // left → right
            let mut x = if row == 0 { mx } else { x_left };
            if row == 0 {
                push((x, y), short);
            }
            while x + sx < x_right - 1.0 {
                x += sx;
                push((x, y), short);
            }
            push((x_right, y), edge);
            if row < 2 {
                push((x_right, y + sy), short); // step down
            }
        } else {
            // right → left
            let mut x = x_right;
            while x - sx > x_left + 1.0 {
                x -= sx;
                push((x, y), short);
            }
            push((x_left, y), edge);
            push((x_left, y + sy), short); // step down
        }
    }
    push((x_right, y_bottom), edge); // below the board: end of the map
    (canvas, path, pauses)
}

impl CameraSource for ViewportSource {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn next_frame(&mut self) -> Result<Option<RgbImage>> {
        if let Some(p) = &mut self.pacer {
            p.wait();
        }
        // Advance along the path
        if self.pause_left > 0 {
            self.pause_left -= 1;
        } else if self.target < self.path.len() {
            let (tx, ty) = self.path[self.target];
            let (dx, dy) = (tx - self.pos.0, ty - self.pos.1);
            let d = dx.hypot(dy);
            if d <= self.speed {
                self.pos = (tx, ty);
                self.pause_left = self.pauses[self.target];
                self.target += 1;
            } else {
                self.pos = (self.pos.0 + dx / d * self.speed, self.pos.1 + dy / d * self.speed);
            }
        } else {
            return Ok(None);
        }
        let (x, y) = (self.pos.0.round() as u32, self.pos.1.round() as u32);
        let crop = image::imageops::crop_imm(&self.board, x, y, self.view.0, self.view.1).to_image();
        Ok(Some(crop))
    }
}

#[cfg(feature = "camera")]
pub mod nokhwa_source {
    use super::*;
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    };
    use nokhwa::Camera;

    /// Lists the detected cameras: `(index, name)`.
    pub fn list_cameras() -> Result<Vec<(u32, String)>> {
        let infos = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
        Ok(infos
            .into_iter()
            .enumerate()
            .map(|(i, c)| (i as u32, c.human_name()))
            .collect())
    }

    pub struct NokhwaSource {
        cam: Camera,
        label: String,
    }

    impl NokhwaSource {
        pub fn open(index: u32, width: u32, height: u32, fps: u32, mjpeg: bool) -> Result<Self> {
            let fmt = CameraFormat::new(
                Resolution::new(width, height),
                if mjpeg { FrameFormat::MJPEG } else { FrameFormat::YUYV },
                fps,
            );
            let req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(fmt));
            let mut cam = Camera::new(CameraIndex::Index(index), req)?;
            cam.open_stream()?;
            let f = cam.camera_format();
            let label = format!(
                "camera {index} ({}x{} {:?} @{}fps)",
                f.resolution().width(),
                f.resolution().height(),
                f.format(),
                f.frame_rate()
            );
            Ok(Self { cam, label })
        }
    }

    impl CameraSource for NokhwaSource {
        fn name(&self) -> String {
            self.label.clone()
        }
        fn next_frame(&mut self) -> Result<Option<RgbImage>> {
            let buf = self.cam.frame()?;
            let rgb = buf.decode_image::<RgbFormat>()?;
            let (w, h) = (rgb.width(), rgb.height());
            Ok(Some(RgbImage::from_raw(w, h, rgb.into_raw()).context("inconsistent image size")?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serpentine_covers_board_and_alternates() {
        let pts = serpentine_path((3200, 2000), (1280, 720), 0.68);
        assert!(pts.len() >= 6);
        assert_eq!(pts[0], (0.0, 0.0));
        // Row 2 is traversed in the opposite direction
        let row0_end = pts.iter().position(|p| p.1 > 0.0).unwrap();
        assert!(pts[row0_end - 1].0 > pts[0].0);
        assert_eq!(pts[row0_end].0, pts[row0_end - 1].0);
        // All points are inside the board
        assert!(pts.iter().all(|p| p.0 + 1280.0 <= 3200.0 && p.1 + 720.0 <= 2000.0));
    }

    #[test]
    fn viewport_source_ends() {
        let board = synthetic_board(800, 500);
        let mut s = ViewportSource::new(board, (400, 300), 0.7, 40.0, 1, None);
        let mut n = 0;
        while let Some(f) = s.next_frame().unwrap() {
            assert_eq!((f.width(), f.height()), (400, 300));
            n += 1;
            assert!(n < 10_000);
        }
        assert!(s.is_done());
    }
}
