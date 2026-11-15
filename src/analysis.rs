//! Frame analysis: motion, position, sharpness, capture decision, guidance.

use crate::capture::{CaptureMachine, Input, Params, State};
use crate::edge::{Band, BlockMap, EdgeDetector, EdgeParams, Side};
use crate::focus::sharpness;
use crate::guidance::{AutoStep, Guide, Phase};
use crate::mosaic::Mosaic;
use crate::session::Session;
use crate::tracking::{gray_small, to_f32, PhaseCorrelator};
use anyhow::Result;
use image::{GrayImage, RgbImage};
use std::collections::VecDeque;

/// Downscale factor for the analysis (÷4 → scale 0.25).
pub const ANALYSIS_K: u32 = 4;

/// Minimum new area (fraction of the field) since the last tile to warrant a closing capture.
pub const CLOSING_MIN_OFFSET: f64 = 0.10;

/// How many recent stable frames compete to become the saved tile (≈ 0.25 s at 30 fps).
const BEST_OF: usize = 8;

/// A recent stable frame, a candidate for the saved tile.
struct Recent {
    frame: RgbImage,
    gray: GrayImage,
    sharp: f32,
    pos: (f64, f64),
}

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
    /// Direction (unit axis vector) the camera should move next, `None` once the map is finished.
    pub heading: Option<(f64, f64)>,
    /// +1 = the current row runs to the right, -1 = to the left.
    pub row_dir: f64,
    pub phase: Phase,
    /// Depth (fraction of the field) of the table seen beyond the board edge ahead, 0 if none.
    pub edge_ahead: f32,
    /// Last automatic step (row ended / map finished), revertible with "not the edge".
    pub auto_step: Option<AutoStep>,
    pub rows_done: usize,
    pub finished: bool,
    pub auto_detect: bool,
}

pub struct Analyzer {
    pc: Option<PhaseCorrelator>,
    prev: Option<Vec<f32>>,
    tiles: Vec<TileRef>,
    pos: (f64, f64),
    last_offset_frac: f64,
    recent: VecDeque<Recent>,
    pub machine: CaptureMachine,
    pub mosaic: Mosaic,
    pub session: Option<Session>,
    pub guide: Guide,
    pub edge: EdgeDetector,
    /// Automatic edge detection; the manual controls always work.
    pub auto_detect: bool,
    /// Table band seen ahead on the last frame.
    ahead: Option<Band>,
    /// Band that triggered the last automatic step.
    fired: Option<Band>,
}

impl Analyzer {
    pub fn new(params: Params, session: Option<Session>) -> Self {
        Self {
            pc: None,
            prev: None,
            tiles: Vec::new(),
            pos: (0.0, 0.0),
            last_offset_frac: 0.0,
            recent: VecDeque::new(),
            machine: CaptureMachine::new(params),
            mosaic: Mosaic::new(),
            session,
            guide: Guide::default(),
            edge: EdgeDetector::new(EdgeParams::default()),
            auto_detect: true,
            ahead: None,
            fired: None,
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

    // ---- guidance controls (manual fallbacks of the automatic edge handling) ----

    /// Is there new area since the last tile that a closing capture would cover?
    fn needs_closing(&self) -> bool {
        !self.tiles.is_empty() && self.last_offset_frac >= CLOSING_MIN_OFFSET
    }

    /// Manual: the row ends here.
    pub fn end_row(&mut self) {
        let nc = self.needs_closing();
        self.guide.end_row(nc);
    }

    /// Manual: the map is complete.
    pub fn finish(&mut self) {
        let nc = self.needs_closing();
        self.guide.finish(nc);
    }

    /// Manual: flip the row direction.
    pub fn reverse(&mut self) {
        self.guide.reverse();
    }

    /// Manual: resume after a finish.
    pub fn resume(&mut self) {
        self.guide.resume();
        self.edge.rearm();
    }

    /// "Not the edge": reverts the last automatic step (or ignores the band seen ahead) and never
    /// fires again on that appearance; the detector stays quiet until the next capture.
    pub fn not_an_edge(&mut self) {
        let band = if self.guide.revert_auto() { self.fired.take() } else { self.ahead };
        if let Some(b) = band {
            self.edge.reject(&b.look);
        }
        self.ahead = None;
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
        let fsz = (frame_size.0 as f64, frame_size.1 as f64);
        let offset = self.tiles.last().map(|tr| (self.pos.0 - tr.pos.0, self.pos.1 - tr.pos.1));
        let off_frac = offset.map_or(0.0, |(dx, dy)| (dx.abs() / fsz.0).max(dy.abs() / fsz.1));
        self.last_offset_frac = off_frac;
        let motion_px = match motion {
            Some((dx, dy)) => dx.hypot(dy),
            None if first => 0.0,
            None => f64::INFINITY,
        };

        // Closing capture: after a confirmed row/map end, grab the last tile as soon as the
        // image is stable and sharp, wherever it is. If there is nothing new, skip it.
        if matches!(self.guide.phase, Phase::Closing(_)) && off_frac < CLOSING_MIN_OFFSET {
            self.guide.closing_done();
        }
        let force_far = matches!(self.guide.phase, Phase::Closing(_));
        let d = self.machine.update(&Input {
            t,
            motion_px,
            tracking_ok: motion.is_some() || tile_ok || first,
            offset,
            frame: fsz,
            sharp,
            force_far,
        });

        // 4) capture: the sharpest of the recent stable frames is saved, not merely the one that
        // happened to trigger (hand tremor and focus breathing make neighbouring frames differ).
        if motion_px < self.machine.params.stable_px {
            if self.recent.len() == BEST_OF {
                self.recent.pop_front();
            }
            // Only worth keeping a copy when a capture is plausible soon.
            self.recent.push_back(Recent { frame: frame.clone(), gray: gray.clone(), sharp, pos: self.pos });
        } else {
            self.recent.clear();
        }
        let capture = d.capture && !self.guide.finished;
        if capture {
            let best = self
                .recent
                .iter()
                .max_by(|a, b| a.sharp.partial_cmp(&b.sharp).unwrap_or(std::cmp::Ordering::Equal))
                .map(|r| (r.frame.clone(), r.gray.clone(), r.pos));
            let (tile_frame, tile_gray, tile_pos) = best.unwrap_or_else(|| (frame.clone(), gray.clone(), self.pos));
            if let Some(s) = &mut self.session {
                s.add_tile(&tile_frame, tile_pos)?;
            }
            self.mosaic.add(&tile_frame, tile_pos);
            // What the board looks like (the middle of every tile) is what the table is not.
            let tile_map = BlockMap::new(&tile_frame, &tile_gray, ANALYSIS_K, self.edge.params.block);
            self.edge.observe_tile(&tile_map);
            self.tiles.push(TileRef { gray: tile_gray, pos: tile_pos });
            self.recent.clear();
            self.guide.tile_captured();
            if self.tiles.len() == 1 && self.guide.rows_done == 0 && self.guide.phase == Phase::Row && self.auto_detect {
                // Started at the right end of the first row: the rows run to the left.
                let right = self.edge.band(&tile_map, Side::Right).is_some();
                let left = self.edge.band(&tile_map, Side::Left).is_some();
                if right && !left {
                    self.guide.row_dir = -1.0;
                }
            }
        }

        // 5) automatic edge handling: table ahead → end of the row (or of the map)
        let watch = self.auto_detect
            && !capture
            && !self.tiles.is_empty()
            && self.guide.edge_meaning().is_some()
            && d.state != State::Capture;
        let status = match self.guide.heading() {
            Some(hd) if watch => {
                let map = BlockMap::new(frame, &gray, ANALYSIS_K, self.edge.params.block);
                self.edge.update(Some(&map), Side::from_heading(hd), t, motion_px)
            }
            _ => self.edge.update(None, Side::Right, t, 0.0),
        };
        self.ahead = status.band;
        if status.edge {
            let nc = self.needs_closing();
            if self.guide.edge_reached(nc).is_some() {
                if let Some(b) = status.band {
                    self.edge.accept(&b);
                    self.fired = Some(b);
                }
            }
        }

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
            captured: capture,
            peak,
            analysis_ms: t0.elapsed().as_secs_f32() * 1000.0,
            heading: self.guide.heading(),
            row_dir: self.guide.row_dir,
            phase: self.guide.phase,
            edge_ahead: self.ahead.map_or(0.0, |b| b.depth),
            auto_step: self.guide.auto_step(),
            rows_done: self.guide.rows_done,
            finished: self.guide.finished,
            auto_detect: self.auto_detect,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{demo_scene, serpentine_path, synthetic_board, CameraSource, ViewportSource};

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

    /// The demo hand overshoots every board edge onto a wood-grain table: with no one pressing
    /// anything, the analyzer must end each row, capture its last tile, step down, and finish
    /// when the step down shows the table below.
    #[test]
    fn demo_run_is_fully_automatic() {
        let (canvas, path, pauses) = demo_scene();
        let mut src = ViewportSource::with_path(canvas, (1280, 720), path, pauses, 14.0, None);
        let mut an = Analyzer::new(Params::default(), None);
        let (mut i, mut edge_frames, mut autos, mut last) = (0u32, 0u32, Vec::new(), None::<Report>);
        while let Some(f) = src.next_frame().unwrap() {
            let r = an.process(&f, i as f64 / 30.0).unwrap();
            if r.edge_ahead > 0.0 {
                edge_frames += 1;
            }
            if r.auto_step.is_some() && last.as_ref().is_some_and(|l| l.auto_step != r.auto_step) {
                autos.push((i, r.auto_step.unwrap(), r.rows_done));
            }
            if r.captured {
                eprintln!("i={i} tile #{} at ({:.0},{:.0}) phase={:?} rows={}", r.tiles, r.pos.0, r.pos.1, r.phase, r.rows_done);
            }
            last = Some(r);
            i += 1;
        }
        let r = last.unwrap();
        eprintln!("{i} frames, {} tiles, {} rows, finished={}, edge frames={edge_frames}, autos={autos:?}", r.tiles, r.rows_done, r.finished);
        assert!(r.finished, "the map was not finished");
        assert_eq!(r.rows_done, 3);
        assert!(r.tiles >= 12, "only {} tiles", r.tiles);
        assert_eq!(autos.iter().filter(|a| a.1 == AutoStep::RowEnded).count(), 3, "{autos:?}");
        assert_eq!(autos.last().map(|a| a.1), Some(AutoStep::MapFinished));
    }

    /// "Not the edge" right after an automatic row end puts the guide back on the row, and the
    /// same table does not end the row again.
    #[test]
    fn not_an_edge_reverts_and_sticks() {
        let (canvas, path, pauses) = demo_scene();
        let mut src = ViewportSource::with_path(canvas, (1280, 720), path, pauses, 14.0, None);
        let mut an = Analyzer::new(Params::default(), None);
        let mut i = 0u32;
        let mut reverted_at = None;
        while let Some(f) = src.next_frame().unwrap() {
            let r = an.process(&f, i as f64 / 30.0).unwrap();
            i += 1;
            if let Some(at) = reverted_at {
                assert_ne!(r.auto_step, Some(AutoStep::RowEnded), "row ended again {} frames later", i - at);
                if i - at > 200 {
                    break;
                }
            } else if r.auto_step == Some(AutoStep::RowEnded) {
                an.not_an_edge();
                assert_eq!(an.guide.phase, Phase::Row);
                assert_eq!(an.guide.rows_done, 0);
                assert_eq!(an.guide.heading(), Some((1.0, 0.0)));
                reverted_at = Some(i);
            }
        }
        assert!(reverted_at.is_some(), "the row end was never detected");
    }

    /// Starting at the right end of the first row (table on the right): rows run to the left.
    #[test]
    fn first_tile_at_the_right_edge_flips_the_rows() {
        let (canvas, _, _) = demo_scene();
        let (mx, my, bw) = (560.0, 320.0, 2700.0);
        let x = mx + bw - 0.7 * 1280.0; // 30 % of the view beyond the right edge
        let path = vec![(x, my), (x - 500.0, my)];
        let mut src = ViewportSource::with_path(canvas, (1280, 720), path, vec![20, 20], 14.0, None);
        let mut an = Analyzer::new(Params::default(), None);
        let mut i = 0;
        while an.tile_count() < 1 {
            let f = src.next_frame().unwrap().expect("stream ended too early");
            an.process(&f, i as f64 / 30.0).unwrap();
            i += 1;
        }
        assert_eq!(an.guide.heading(), Some((-1.0, 0.0)));
    }

    /// Detection off: nothing advances by itself; the manual fallbacks work.
    #[test]
    fn manual_controls_drive_the_guide() {
        let (canvas, path, pauses) = demo_scene();
        let mut src = ViewportSource::with_path(canvas, (1280, 720), path, pauses, 14.0, None);
        let mut an = Analyzer::new(Params::default(), None);
        an.auto_detect = false; // fully manual: no suggestion may ever appear
        let mut i = 0u32;
        let mut ended_row = false;
        while let Some(f) = src.next_frame().unwrap() {
            let r = an.process(&f, i as f64 / 30.0).unwrap();
            assert_eq!(r.edge_ahead, 0.0);
            assert!(r.auto_step.is_none());
            // At the first row end (long pause, offset since the last tile), the user presses "end row"
            // (the state reads `Move` there: the offset is below the 65 % trigger, so only stability counts)
            if !ended_row && r.tiles >= 3 && r.offset_frac > 0.1 && r.stable_secs > 1.0 {
                an.end_row();
                ended_row = true;
            }
            i += 1;
        }
        assert!(ended_row, "the row end was never reached");
        assert!(an.guide.rows_done >= 1, "the manual end-of-row did not advance the guide");
        assert!(an.tile_count() >= 5, "the closing capture and the next row were not captured");
    }

    /// Among the frames of a stable stretch, the sharpest one is what gets saved.
    #[test]
    fn the_sharpest_recent_frame_is_saved() {
        use crate::focus::sharpness;
        use crate::tracking::gray_small;
        use image::imageops;

        let board = synthetic_board(900, 500);
        let sharp = imageops::crop_imm(&board, 100, 60, 640, 360).to_image();
        let soft = imageops::blur(&sharp, 1.2);
        // frame = (1 - a) sharp + a soft, so neighbours differ in sharpness but stay close to it
        let mix = |a: f32| {
            let mut f = sharp.clone();
            for (o, (p, q)) in f.pixels_mut().zip(sharp.pixels().zip(soft.pixels())) {
                for c in 0..3 {
                    o.0[c] = (p.0[c] as f32 * (1.0 - a) + q.0[c] as f32 * a).round() as u8;
                }
            }
            f
        };
        let (best, dull) = (mix(0.0), mix(0.25));
        let (s_best, s_dull) = (sharpness(&gray_small(&best, 4)), sharpness(&gray_small(&dull, 4)));
        assert!(s_dull / s_best > 0.8 && s_dull < s_best * 0.97, "test frames: {s_dull} vs {s_best}");

        let root = std::env::temp_dir().join(format!("pcbmapper-bestof-{}", std::process::id()));
        let session = Session::create(&root).unwrap();
        let mut an = Analyzer::new(Params::default(), Some(session));
        let mut captured = false;
        for i in 0..20u32 {
            let frame = if i == 4 { &best } else { &dull };
            let r = an.process(frame, i as f64 / 30.0).unwrap();
            captured |= r.captured;
            if r.captured {
                assert!(i > 4, "captured before the sharp frame was seen");
                break;
            }
        }
        assert!(captured, "no capture happened");
        let dir = an.session.as_ref().unwrap().dir().to_path_buf();
        let saved = image::open(dir.join("tiles/00001.png")).unwrap().to_rgb8();
        let s_saved = sharpness(&gray_small(&saved, 4));
        assert!((s_saved - s_best).abs() < 0.01 * s_best, "saved {s_saved}, sharpest {s_best}, dull {s_dull}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
