//! State machine that triggers the capture.
//!
//! `Tracking` → `Stabilizing` → `Capture` → `Move`; `Lost` if tracking fails.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Tracking is active but the image is soft or no condition is met yet.
    Tracking,
    /// Offset reached, waiting for the image to be stable.
    Stabilizing,
    /// All conditions met: capture triggered on this frame.
    Capture,
    /// Capture done: move the board until the next offset.
    Move,
    /// Correlation lost: go back to an area already captured.
    Lost,
}

#[derive(Debug, Clone)]
pub struct Params {
    /// Maximum motion (full-resolution px per frame) to count as "stable".
    pub stable_px: f64,
    /// Required stability duration (seconds).
    pub stable_secs: f64,
    /// Offset since the last capture, as a fraction of the field (0.65 → 35 % overlap).
    pub trigger_frac: f64,
    /// Minimum sharpness relative to the best recent score.
    pub min_sharp_ratio: f32,
    /// History window for the "best recent score" (seconds).
    pub sharp_window_secs: f64,
    /// Tracking-failure duration before switching to `Lost` (seconds).
    pub lost_secs: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            stable_px: 1.5,
            stable_secs: 0.25,
            trigger_frac: 0.65,
            min_sharp_ratio: 0.8,
            sharp_window_secs: 1.5,
            lost_secs: 0.4,
        }
    }
}

pub struct Input {
    /// Current time (seconds).
    pub t: f64,
    /// Motion magnitude since the previous frame (full-resolution px); `INFINITY` if unknown.
    pub motion_px: f64,
    /// Is tracking (frame-to-frame or vs the last tile) reliable?
    pub tracking_ok: bool,
    /// Camera offset since the last capture (px), `None` if nothing was captured yet.
    pub offset: Option<(f64, f64)>,
    /// Field size (full-resolution px).
    pub frame: (f64, f64),
    /// Sharpness of the current frame.
    pub sharp: f32,
    /// Capture as soon as stable and sharp, whatever the offset (closing capture of a row/map).
    pub force_far: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub state: State,
    pub capture: bool,
    pub stable_secs: f64,
    /// Sharpness relative to the best recent score (1 = at best).
    pub sharp_rel: f32,
    /// Offset since the last capture as a fraction of the field (0 if nothing captured).
    pub offset_frac: f64,
}

pub struct CaptureMachine {
    pub params: Params,
    sharp_hist: VecDeque<(f64, f32)>,
    stable_since: Option<f64>,
    bad_since: Option<f64>,
}

impl CaptureMachine {
    pub fn new(params: Params) -> Self {
        Self { params, sharp_hist: VecDeque::new(), stable_since: None, bad_since: None }
    }

    /// Resets stability (after a capture or an undo).
    pub fn reset_stability(&mut self) {
        self.stable_since = None;
    }

    pub fn update(&mut self, i: &Input) -> Decision {
        let p = &self.params;

        // Sharpness history
        self.sharp_hist.push_back((i.t, i.sharp));
        while self.sharp_hist.front().is_some_and(|(t, _)| i.t - t > p.sharp_window_secs) {
            self.sharp_hist.pop_front();
        }
        let best = self.sharp_hist.iter().map(|&(_, s)| s).fold(0.0f32, f32::max);
        let sharp_rel = if best > 0.0 { i.sharp / best } else { 1.0 };

        // Tracking loss (only once a reference tile exists)
        if i.tracking_ok {
            self.bad_since = None;
        } else {
            self.bad_since.get_or_insert(i.t);
        }
        let lost = i.offset.is_some() && self.bad_since.is_some_and(|t0| i.t - t0 >= p.lost_secs);

        // Stability
        if i.motion_px < p.stable_px {
            self.stable_since.get_or_insert(i.t);
        } else {
            self.stable_since = None;
        }
        let stable_secs = self.stable_since.map_or(0.0, |t0| i.t - t0);

        let offset_frac = i.offset.map_or(0.0, |(dx, dy)| (dx.abs() / i.frame.0).max(dy.abs() / i.frame.1));
        let far_enough = i.offset.is_none() || i.force_far || offset_frac >= p.trigger_frac;

        let mut capture = false;
        let state = if lost {
            State::Lost
        } else if !far_enough {
            State::Move
        } else if stable_secs < p.stable_secs {
            State::Stabilizing
        } else if sharp_rel < p.min_sharp_ratio {
            State::Tracking // stable but soft image: adjust focus
        } else {
            capture = true;
            State::Capture
        };

        if capture {
            self.stable_since = None;
            self.sharp_hist.clear();
        }
        Decision { state, capture, stable_secs, sharp_rel, offset_frac }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(t: f64, motion: f64, offset: Option<(f64, f64)>, sharp: f32) -> Input {
        Input { t, motion_px: motion, tracking_ok: true, offset, frame: (1000.0, 600.0), sharp, force_far: false }
    }

    #[test]
    fn first_capture_needs_stability() {
        let mut m = CaptureMachine::new(Params::default());
        let d = m.update(&input(0.0, 0.0, None, 100.0));
        assert_eq!(d.state, State::Stabilizing);
        assert!(!d.capture);
        let d = m.update(&input(0.1, 0.0, None, 100.0));
        assert!(!d.capture);
        let d = m.update(&input(0.3, 0.0, None, 100.0));
        assert!(d.capture && d.state == State::Capture);
    }

    #[test]
    fn motion_resets_stability() {
        let mut m = CaptureMachine::new(Params::default());
        m.update(&input(0.0, 0.0, None, 100.0));
        m.update(&input(0.2, 10.0, None, 100.0)); // motion
        let d = m.update(&input(0.3, 0.0, None, 100.0));
        assert!(!d.capture); // stability restarts from 0.3
        let d = m.update(&input(0.6, 0.0, None, 100.0));
        assert!(d.capture);
    }

    #[test]
    fn after_capture_wait_for_offset() {
        let mut m = CaptureMachine::new(Params::default());
        m.update(&input(0.0, 0.0, None, 100.0));
        assert!(m.update(&input(0.3, 0.0, None, 100.0)).capture);
        // Small offset: the board must keep moving
        let d = m.update(&input(0.4, 12.0, Some((300.0, 0.0)), 100.0));
        assert_eq!(d.state, State::Move);
        assert!(!d.capture);
        // Enough offset (≥ 65 % of 1000 px) but not stable yet
        let d = m.update(&input(1.0, 12.0, Some((680.0, 0.0)), 100.0));
        assert_eq!(d.state, State::Stabilizing);
        let d = m.update(&input(1.1, 0.0, Some((680.0, 0.0)), 100.0));
        assert_eq!(d.state, State::Stabilizing);
        let d = m.update(&input(1.4, 0.0, Some((680.0, 0.0)), 100.0));
        assert!(d.capture);
    }

    #[test]
    fn force_far_captures_without_offset() {
        let mut m = CaptureMachine::new(Params::default());
        m.update(&input(0.0, 0.0, None, 100.0));
        assert!(m.update(&input(0.3, 0.0, None, 100.0)).capture);
        let mut i = input(1.0, 0.0, Some((120.0, 0.0)), 100.0);
        assert!(!m.update(&i).capture);
        i.t = 1.3;
        assert!(!m.update(&i).capture);
        i.t = 1.4;
        i.force_far = true;
        assert!(m.update(&i).capture);
    }

    #[test]
    fn blur_blocks_capture() {
        let mut m = CaptureMachine::new(Params::default());
        m.update(&input(0.0, 0.0, None, 100.0)); // best score = 100
        let d = m.update(&input(0.3, 0.0, None, 50.0)); // stable but soft
        assert_eq!(d.state, State::Tracking);
        assert!(!d.capture);
        let d = m.update(&input(0.4, 0.0, None, 95.0));
        assert!(d.capture);
    }

    #[test]
    fn lost_after_tracking_failure() {
        let mut m = CaptureMachine::new(Params::default());
        let mut inp = input(0.0, 0.0, Some((100.0, 0.0)), 100.0);
        inp.tracking_ok = false;
        assert_ne!(m.update(&inp).state, State::Lost);
        inp.t = 0.5;
        assert_eq!(m.update(&inp).state, State::Lost);
        inp.t = 0.6;
        inp.tracking_ok = true;
        assert_ne!(m.update(&inp).state, State::Lost);
    }
}
