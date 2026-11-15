//! Automatic board-edge detection.
//!
//! When the view reaches the edge of the PCB, the side of the image the user is heading toward
//! shows the table: a flat, texture-less area. The detector measures the share of low-variance
//! blocks in that strip. Large flat areas also exist on real boards (ground planes, solder mask),
//! so a detection is only a *suggestion*: the user confirms or dismisses it.

use image::GrayImage;

#[derive(Debug, Clone)]
pub struct EdgeParams {
    /// Block size (px, in the analysis image) on which the variance is measured.
    pub block: u32,
    /// A block is "flat" when its standard deviation (gray levels) is below this.
    pub std_thr: f32,
    /// Width of the "ahead" strip, as a fraction of the image.
    pub strip: f32,
    /// Share of flat blocks required in the strip.
    pub ahead_uniform: f32,
    /// If more than this share of the whole image is flat, it is not an edge (lens cap, blank sheet).
    pub max_total_uniform: f32,
    /// The condition must hold this long (seconds) before suggesting.
    pub hold_secs: f64,
    /// A suggestion is withdrawn after the condition has been false this long (seconds).
    pub clear_secs: f64,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            block: 8,
            std_thr: 3.0,
            strip: 0.30,
            ahead_uniform: 0.90,
            max_total_uniform: 0.85,
            hold_secs: 0.4,
            clear_secs: 2.0,
        }
    }
}

/// Share of flat blocks `(in the strip ahead, in the whole image)`.
/// `heading` is the camera's direction of travel (unit axis vector).
pub fn flat_fractions(gray: &GrayImage, heading: (f64, f64), p: &EdgeParams) -> (f32, f32) {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let b = p.block as usize;
    let (bw, bh) = (w / b, h / b);
    if bw == 0 || bh == 0 {
        return (0.0, 0.0);
    }
    let px = gray.as_raw();
    let thr2 = (p.std_thr * p.std_thr) as f64;
    let n = (b * b) as f64;
    let strip_x = (bw as f32 * p.strip).ceil() as usize;
    let strip_y = (bh as f32 * p.strip).ceil() as usize;
    let ahead = |bx: usize, by: usize| -> bool {
        if heading.0 > 0.5 {
            bx >= bw.saturating_sub(strip_x)
        } else if heading.0 < -0.5 {
            bx < strip_x
        } else if heading.1 > 0.5 {
            by >= bh.saturating_sub(strip_y)
        } else {
            by < strip_y
        }
    };
    let (mut a_tot, mut a_flat, mut tot, mut flat) = (0u32, 0u32, 0u32, 0u32);
    for by in 0..bh {
        for bx in 0..bw {
            let (mut s, mut s2) = (0.0f64, 0.0f64);
            for y in 0..b {
                let row = (by * b + y) * w + bx * b;
                for x in 0..b {
                    let v = px[row + x] as f64;
                    s += v;
                    s2 += v * v;
                }
            }
            let var = s2 / n - (s / n) * (s / n);
            let is_flat = var < thr2;
            tot += 1;
            flat += is_flat as u32;
            if ahead(bx, by) {
                a_tot += 1;
                a_flat += is_flat as u32;
            }
        }
    }
    (a_flat as f32 / a_tot.max(1) as f32, flat as f32 / tot.max(1) as f32)
}

/// Turns the per-frame measure into a stable, latched suggestion.
pub struct EdgeDetector {
    pub params: EdgeParams,
    since: Option<f64>,
    clear_since: Option<f64>,
    suggested_at: Option<f64>,
    dismissed: bool,
}

impl EdgeDetector {
    pub fn new(params: EdgeParams) -> Self {
        Self { params, since: None, clear_since: None, suggested_at: None, dismissed: false }
    }

    /// Time (seconds) at which the current suggestion appeared, if any.
    pub fn suggested_at(&self) -> Option<f64> {
        self.suggested_at
    }

    /// Updates with a new frame; returns whether an edge is currently suggested.
    pub fn update(&mut self, gray: &GrayImage, heading: (f64, f64), t: f64, enabled: bool) -> bool {
        if !enabled {
            self.clear();
            return false;
        }
        let (ahead, total) = flat_fractions(gray, heading, &self.params);
        let cond = ahead >= self.params.ahead_uniform && total <= self.params.max_total_uniform;
        if cond {
            self.clear_since = None;
            let s = *self.since.get_or_insert(t);
            if t - s >= self.params.hold_secs && !self.dismissed && self.suggested_at.is_none() {
                self.suggested_at = Some(t);
            }
        } else {
            self.since = None;
            let c = *self.clear_since.get_or_insert(t);
            if self.suggested_at.is_some() && t - c >= self.params.clear_secs {
                self.suggested_at = None;
            }
        }
        self.suggested_at.is_some()
    }

    /// The user dismissed or confirmed: no new suggestion until the next capture.
    pub fn dismiss(&mut self) {
        self.suggested_at = None;
        self.since = None;
        self.dismissed = true;
    }

    /// A tile was captured (the user moved on): re-arm the detector.
    pub fn rearm(&mut self) {
        self.clear();
        self.dismissed = false;
    }

    /// Nothing to detect right now (disabled, no reference tile, finished…): drop any suggestion.
    pub fn idle(&mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        self.since = None;
        self.clear_since = None;
        self.suggested_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    /// 240x136 image: textured on the left, flat on the right of `edge_x`.
    fn scene(edge_x: u32) -> GrayImage {
        GrayImage::from_fn(240, 136, |x, y| {
            if x < edge_x {
                Luma([(((x * 37 + y * 91) % 97) as u8).saturating_mul(2)]) // busy texture
            } else {
                Luma([60 + ((x + y) % 2) as u8]) // flat table with tiny noise
            }
        })
    }

    #[test]
    fn flat_strip_detected_only_in_heading_direction() {
        let img = scene(150); // flat area covers the right 37 %
        let p = EdgeParams::default();
        let (ahead_r, total) = flat_fractions(&img, (1.0, 0.0), &p);
        let (ahead_l, _) = flat_fractions(&img, (-1.0, 0.0), &p);
        assert!(ahead_r > 0.9, "{ahead_r}");
        assert!(ahead_l < 0.1, "{ahead_l}");
        assert!(total < 0.5, "{total}");
    }

    #[test]
    fn suggestion_needs_hold_time_and_can_be_dismissed() {
        let img = scene(150);
        let mut d = EdgeDetector::new(EdgeParams::default());
        assert!(!d.update(&img, (1.0, 0.0), 0.0, true));
        assert!(!d.update(&img, (1.0, 0.0), 0.2, true));
        assert!(d.update(&img, (1.0, 0.0), 0.5, true));
        assert!(d.suggested_at().is_some());
        d.dismiss();
        assert!(!d.update(&img, (1.0, 0.0), 1.0, true)); // stays quiet after dismissal
        assert!(!d.update(&img, (1.0, 0.0), 3.0, true));
        d.rearm();
        d.update(&img, (1.0, 0.0), 4.0, true);
        assert!(d.update(&img, (1.0, 0.0), 4.6, true)); // back after a capture
    }

    #[test]
    fn busy_or_fully_flat_images_never_suggest() {
        let mut d = EdgeDetector::new(EdgeParams::default());
        let busy = scene(240);
        let flat = scene(0);
        for i in 0..20 {
            assert!(!d.update(&busy, (1.0, 0.0), i as f64 * 0.1, true));
            assert!(!d.update(&flat, (1.0, 0.0), 10.0 + i as f64 * 0.1, true));
        }
    }

    #[test]
    fn suggestion_expires_when_condition_clears() {
        let mut d = EdgeDetector::new(EdgeParams::default());
        let edge = scene(150);
        let busy = scene(240);
        d.update(&edge, (1.0, 0.0), 0.0, true);
        assert!(d.update(&edge, (1.0, 0.0), 0.5, true));
        assert!(d.update(&busy, (1.0, 0.0), 1.0, true)); // still latched for a moment
        assert!(!d.update(&busy, (1.0, 0.0), 3.5, true)); // withdrawn after clear_secs
    }

    #[test]
    fn disabled_detector_is_silent() {
        let mut d = EdgeDetector::new(EdgeParams::default());
        let edge = scene(150);
        d.update(&edge, (1.0, 0.0), 0.0, true);
        assert!(!d.update(&edge, (1.0, 0.0), 1.0, false));
    }
}
