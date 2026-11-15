//! Frame analysis: motion, position, sharpness, capture decision.

use crate::capture::{CaptureMachine, Input, Params, State};
use crate::focus::sharpness;
use crate::mosaic::Mosaic;
use crate::session::Session;
use crate::tracking::{gray_small, to_f32, PhaseCorrelator};
use anyhow::Result;
use image::{GrayImage, RgbImage};

/// Downscale factor for the analysis (÷4 → scale 0.25).
pub const ANALYSIS_K: u32 = 4;

/// Reference tile for absolute tracking.
struct TileRef {
    gray: GrayImage,
    pos: (f64, f64),
}

/// Result of a frame analysis, sent to the UI.
#[derive(Debug, Clone)]
pub struct Report {
    pub state: State,
    /// Motion since the previous frame (full-resolution px), `None` if unreliable.
    pub motion: Option<(f64, f64)>,
    /// Estimated camera position (top-left corner of the field, board coordinates, px).
    pub pos: (f64, f64),
    pub sharp: f32,
    pub sharp_rel: f32,
    pub stable_secs: f64,
    pub offset_frac: f64,
    pub tiles: usize,
    pub frame_size: (u32, u32),
    /// A tile was just captured on this frame.
    pub captured: bool,
    /// Quality of the frame-to-frame correlation peak.
    pub peak: f32,
    pub analysis_ms: f32,
    /// Direction (unit axis vector) the camera should move next, in board coordinates.
    /// Derived from the last two tiles; defaults to +x.
    pub heading: (f64, f64),
}

pub struct Analyzer {
    pc: Option<PhaseCorrelator>,
    prev: Option<Vec<f32>>,
    tiles: Vec<TileRef>,
    pos: (f64, f64),
    pub machine: CaptureMachine,
    pub mosaic: Mosaic,
    pub session: Option<Session>,
}

impl Analyzer {
    pub fn new(params: Params, session: Option<Session>) -> Self {
        Self {
            pc: None,
            prev: None,
            tiles: Vec::new(),
            pos: (0.0, 0.0),
            machine: CaptureMachine::new(params),
            mosaic: Mosaic::new(),
            session,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Undoes the last tile: mosaic, session and reference tile roll back.
    pub fn undo_last(&mut self) -> Result<bool> {
        if self.tiles.pop().is_none() {
            return Ok(false);
        }
        self.mosaic.pop();
        if let Some(s) = &mut self.session {
            s.remove_last()?;
        }
        self.machine.reset_stability();
        Ok(true)
    }

    pub fn process(&mut self, frame: &RgbImage, t: f64) -> Result<Report> {
        let t0 = std::time::Instant::now();
        let k = ANALYSIS_K as f64;
        let gray = gray_small(frame, ANALYSIS_K);
        let (w, h) = (gray.width() as usize, gray.height() as usize);
        let cur = to_f32(&gray);
        let sharp = sharpness(&gray);

        if self.pc.as_ref().map(|p| p.size()) != Some((w, h)) {
            self.pc = Some(PhaseCorrelator::new(w, h));
            self.prev = None;
        }
        let pc = self.pc.as_mut().expect("correlator initialised above");

        // 1) instantaneous motion (previous frame)
        let (mut motion, mut peak) = (None, 0.0);
        let first = self.prev.is_none();
        if let Some(prev) = &self.prev {
            let s = pc.correlate(prev, &cur);
            peak = s.peak;
            if s.is_reliable() {
                motion = Some((s.dx * k, s.dy * k));
            }
        }
        if let Some((dx, dy)) = motion {
            // Content moves by +d in the image ⇒ the camera moves by -d.
            self.pos = (self.pos.0 - dx, self.pos.1 - dy);
        }

        // 2) absolute position: correlation with the last captured tile
        let mut tile_ok = false;
        if let Some(tr) = &self.tiles.last() {
            let s = pc.correlate(&to_f32(&tr.gray), &cur);
            if s.is_reliable() {
                self.pos = (tr.pos.0 - s.dx * k, tr.pos.1 - s.dy * k);
                tile_ok = true;
            }
        }
        self.prev = Some(cur);

        // 3) decision
        let frame_size = (frame.width(), frame.height());
        let offset = self.tiles.last().map(|tr| (self.pos.0 - tr.pos.0, self.pos.1 - tr.pos.1));
        let motion_px = match motion {
            Some((dx, dy)) => dx.hypot(dy),
            None if first => 0.0,
            None => f64::INFINITY,
        };
        let d = self.machine.update(&Input {
            t,
            motion_px,
            tracking_ok: motion.is_some() || tile_ok || first,
            offset,
            frame: (frame_size.0 as f64, frame_size.1 as f64),
            sharp,
        });

        // 4) capture
        if d.capture {
            if let Some(s) = &mut self.session {
                s.add_tile(frame, self.pos)?;
            }
            self.mosaic.add(frame, self.pos);
            self.tiles.push(TileRef { gray, pos: self.pos });
        }

        let n = self.tiles.len();
        let heading = if n >= 2 {
            let (a, b) = (self.tiles[n - 2].pos, self.tiles[n - 1].pos);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            if dx.abs() >= dy.abs() { (dx.signum(), 0.0) } else { (0.0, dy.signum()) }
        } else {
            (1.0, 0.0)
        };

        Ok(Report {
            state: d.state,
            motion,
            pos: self.pos,
            sharp,
            sharp_rel: d.sharp_rel,
            stable_secs: d.stable_secs,
            offset_frac: d.offset_frac,
            tiles: self.tiles.len(),
            frame_size,
            captured: d.capture,
            peak,
            analysis_ms: t0.elapsed().as_secs_f32() * 1000.0,
            heading,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{serpentine_path, synthetic_board, CameraSource, ViewportSource};

    /// Full run over the synthetic board (dimensions chosen so that the last step
    /// of each row/column also exceeds the 65 % threshold): the machine must capture about
    /// one tile per waypoint, at the right relative positions.
    #[test]
    fn serpentine_run_captures_tiles_at_expected_positions() {
        let board = synthetic_board(2265, 1274);
        let view = (960u32, 540u32);
        let path = serpentine_path((2265, 1274), view, 0.68);
        let mut src = ViewportSource::new(board, view, 0.68, 14.0, 15, None);
        let mut an = Analyzer::new(Params::default(), None);

        let mut i = 0u32;
        let mut max_ms = 0.0f32;
        while let Some(f) = src.next_frame().unwrap() {
            let r = an.process(&f, i as f64 / 30.0).unwrap();
            max_ms = max_ms.max(r.analysis_ms);
            if r.captured || r.state == State::Lost {
                eprintln!("i={i} {:?} pos=({:.1},{:.1}) frac={:.2}", r.state, r.pos.0, r.pos.1, r.offset_frac);
            }
            i += 1;
        }
        let tiles = an.mosaic.tiles();
        eprintln!("{} tiles for {} waypoints; max analysis {max_ms:.1} ms", tiles.len(), path.len());
        assert_eq!(tiles.len(), path.len(), "one tile per waypoint");

        // Positions relative to the first tile ≈ real path
        let (ox, oy) = (tiles[0].pos.0 - path[0].0, tiles[0].pos.1 - path[0].1);
        for (t, p) in tiles.iter().zip(&path) {
            let (ex, ey) = (t.pos.0 - ox - p.0, t.pos.1 - oy - p.1);
            assert!(ex.abs() < 6.0 && ey.abs() < 6.0, "deviation {ex:.1},{ey:.1} px at {p:?}");
        }
    }

    #[test]
    fn undo_removes_last_tile() {
        let board = synthetic_board(1200, 800);
        let mut src = ViewportSource::new(board, (600, 400), 0.68, 14.0, 15, None);
        let mut an = Analyzer::new(Params::default(), None);
        let mut i = 0;
        while an.tile_count() < 2 {
            let f = src.next_frame().unwrap().expect("stream ended too early");
            an.process(&f, i as f64 / 30.0).unwrap();
            i += 1;
        }
        assert!(an.undo_last().unwrap());
        assert_eq!(an.tile_count(), 1);
        assert_eq!(an.mosaic.len(), 1);
    }
}
