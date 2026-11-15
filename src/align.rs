//! Per-tile poses: a small rotation (and scale) on top of the translation registration.
//!
//! A board moved by hand under the microscope turns a little between shots. A degree is enough to
//! shift the far edge of a 1920 px tile by 17 px, which no choice of translations can absorb: the
//! seams ghost. Here every tile gets a similarity transform, fitted to many local point matches
//! across all overlaps at once (a linear least-squares problem, solved robustly).

use crate::photometry::pcg;

/// Placement of a tile on the board: `board = R · (p − half) + c`, with `p` in tile pixels
/// (capture resolution, pixel centres at `+0.5`), `half` the tile's half size and
/// `R = [[1 + a, −b], [b, 1 + a]]` (a rotation by ≈ `b` radians, scaled by ≈ `1 + a`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub a: f64,
    pub b: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Pose {
    /// A pure translation: the tile's top-left corner at `pos`.
    pub fn from_position(pos: (f64, f64), w: u32, h: u32) -> Self {
        Self { a: 0.0, b: 0.0, cx: pos.0 + w as f64 / 2.0, cy: pos.1 + h as f64 / 2.0 }
    }

    /// Board point of the tile point `p` given relative to the tile's centre.
    pub fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        let (m, b) = (1.0 + self.a, self.b);
        (m * p.0 - b * p.1 + self.cx, b * p.0 + m * p.1 + self.cy)
    }

    /// Tile point (relative to the centre) seen at the board point `q`.
    pub fn invert(&self, q: (f64, f64)) -> (f64, f64) {
        let (m, b) = (1.0 + self.a, self.b);
        let det = m * m + b * b;
        let (x, y) = (q.0 - self.cx, q.1 - self.cy);
        ((m * x + b * y) / det, (-b * x + m * y) / det)
    }

    pub fn angle_deg(&self) -> f64 {
        self.b.atan2(1.0 + self.a).to_degrees()
    }

    pub fn scale(&self) -> f64 {
        (1.0 + self.a).hypot(self.b)
    }

    pub fn is_translation(&self) -> bool {
        self.a == 0.0 && self.b == 0.0
    }
}

/// The same board point seen in tiles `i` and `j` (tile coordinates relative to their centres).
#[derive(Debug, Clone, Copy)]
pub struct PointMatch {
    pub i: usize,
    pub j: usize,
    pub pi: (f64, f64),
    pub pj: (f64, f64),
    pub w: f64,
}

/// Outcome of [`solve_poses`].
#[derive(Debug, Clone)]
pub struct PoseFit {
    pub poses: Vec<Pose>,
    /// Weighted RMS distance between matched points before and after (capture px).
    pub rms_before: f64,
    pub rms_after: f64,
    /// Matches kept after outlier rejection.
    pub kept: usize,
}

fn mismatch(poses: &[Pose], m: &PointMatch) -> (f64, f64) {
    let (a, b) = (poses[m.i].apply(m.pi), poses[m.j].apply(m.pj));
    (a.0 - b.0, a.1 - b.1)
}

fn weighted_rms(poses: &[Pose], matches: &[PointMatch], w: &[f64]) -> f64 {
    let (mut s, mut t) = (0.0, 0.0);
    for (m, &wi) in matches.iter().zip(w) {
        let (dx, dy) = mismatch(poses, m);
        s += wi * (dx * dx + dy * dy);
        t += wi;
    }
    if t > 0.0 {
        (s / t).sqrt()
    } else {
        0.0
    }
}

/// Fits one similarity per tile so that matched points coincide, starting from `init`.
///
/// Priors keep the problem well posed: every centre is weakly held to its initial value (this
/// also fixes the global position), rotations are weakly pulled to zero (this fixes the global
/// orientation and holds tiles with few matches), scales strongly so (the magnification of the
/// microscope does not change during a scan). `size` is the typical tile size (px), which sets the
/// lever arm of rotations and scales. Matches disagreeing with the solution are down-weighted
/// (Huber) over a few rounds.
pub fn solve_poses(init: &[Pose], matches: &[PointMatch], size: f64) -> PoseFit {
    let n = init.len();
    let base: Vec<f64> = matches.iter().map(|m| m.w.max(0.0)).collect();
    let rms_before = weighted_rms(init, matches, &base);
    if n == 0 || matches.len() < 8 {
        return PoseFit { poses: init.to_vec(), rms_before, rms_after: rms_before, kept: matches.len() };
    }
    let mut w = base.clone();
    let mut poses = init.to_vec();
    let lever = (size / 2.0).max(1.0);
    for round in 0..4 {
        let total: f64 = w.iter().sum();
        let per_tile = total / n as f64;
        // prior strengths, in "matches per tile" at the tile's lever arm
        let lc = 1e-3 * per_tile;
        let lb = 0.02 * per_tile * lever * lever;
        let la = 2.0 * per_tile * lever * lever;
        // unknowns: [a, b, cx, cy] per tile; residual r = J_i θ_i − J_j θ_j + (p_i − p_j)
        let rows = |m: &PointMatch| -> [([f64; 4], [f64; 4], f64); 2] {
            let ((xi, yi), (xj, yj)) = (m.pi, m.pj);
            [([xi, -yi, 1.0, 0.0], [xj, -yj, 1.0, 0.0], xi - xj), ([yi, xi, 0.0, 1.0], [yj, xj, 0.0, 1.0], yi - yj)]
        };
        let mut rhs = vec![0.0; 4 * n];
        let mut diag = vec![0.0; 4 * n];
        for t in 0..n {
            rhs[4 * t + 2] = lc * init[t].cx;
            rhs[4 * t + 3] = lc * init[t].cy;
            diag[4 * t] = la;
            diag[4 * t + 1] = lb;
            diag[4 * t + 2] = lc;
            diag[4 * t + 3] = lc;
        }
        for (m, &wm) in matches.iter().zip(&w) {
            for (ji, jj, c) in rows(m) {
                for k in 0..4 {
                    rhs[4 * m.i + k] -= wm * ji[k] * c;
                    rhs[4 * m.j + k] += wm * jj[k] * c;
                    diag[4 * m.i + k] += wm * ji[k] * ji[k];
                    diag[4 * m.j + k] += wm * jj[k] * jj[k];
                }
            }
        }
        let apply = |x: &[f64], out: &mut [f64]| {
            for t in 0..n {
                out[4 * t] = la * x[4 * t];
                out[4 * t + 1] = lb * x[4 * t + 1];
                out[4 * t + 2] = lc * x[4 * t + 2];
                out[4 * t + 3] = lc * x[4 * t + 3];
            }
            for (m, &wm) in matches.iter().zip(&w) {
                for (ji, jj, _) in rows(m) {
                    let mut r = 0.0;
                    for k in 0..4 {
                        r += ji[k] * x[4 * m.i + k] - jj[k] * x[4 * m.j + k];
                    }
                    let wr = wm * r;
                    for k in 0..4 {
                        out[4 * m.i + k] += wr * ji[k];
                        out[4 * m.j + k] -= wr * jj[k];
                    }
                }
            }
        };
        let x0: Vec<f64> = poses.iter().flat_map(|p| [p.a, p.b, p.cx, p.cy]).collect();
        let x = pcg(&apply, &rhs, &diag, &x0, 2000);
        poses = (0..n).map(|t| Pose { a: x[4 * t], b: x[4 * t + 1], cx: x[4 * t + 2], cy: x[4 * t + 3] }).collect();
        if round == 3 {
            break;
        }
        let mut res: Vec<f64> = matches
            .iter()
            .map(|m| {
                let (dx, dy) = mismatch(&poses, m);
                dx.hypot(dy)
            })
            .collect();
        let mut sorted = res.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut = (2.0 * sorted[sorted.len() / 2]).max(1.0);
        for ((wi, r), b) in w.iter_mut().zip(res.iter_mut()).zip(&base) {
            // Huber, then a hard cut for gross outliers (repeated patterns, parallax)
            *wi = if *r <= cut {
                *b
            } else if *r <= 4.0 * cut {
                b * cut / *r
            } else {
                0.0
            };
        }
    }
    let kept = w.iter().filter(|v| **v > 0.0).count();
    let rms_after = weighted_rms(&poses, matches, &w);
    let rms_init_same = weighted_rms(init, matches, &w);
    if rms_after >= rms_init_same {
        return PoseFit { poses: init.to_vec(), rms_before, rms_after: rms_init_same, kept };
    }
    PoseFit { poses, rms_before, rms_after, kept }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_round_trip() {
        let p = Pose { a: 0.001, b: -0.017, cx: 500.0, cy: -20.0 };
        let q = p.apply((123.0, -45.0));
        let r = p.invert(q);
        assert!((r.0 - 123.0).abs() < 1e-9 && (r.1 + 45.0).abs() < 1e-9);
        assert!((p.angle_deg() + 0.974).abs() < 0.01, "{}", p.angle_deg());
        let t = Pose::from_position((10.0, 20.0), 100, 50);
        assert_eq!(t.apply((-50.0, -25.0)), (10.0, 20.0));
    }

    /// Three tiles in a row, the middle one turned by 1°: the solve must find the rotation from
    /// the point matches alone, starting from translations.
    #[test]
    fn recovers_a_rotated_tile() {
        let (w, h) = (400.0, 300.0);
        let truth = [
            Pose { a: 0.0, b: 0.0, cx: 200.0, cy: 150.0 },
            Pose { a: 0.0, b: 1f64.to_radians().sin(), cx: 480.0, cy: 160.0 },
            Pose { a: 0.0, b: -0.3f64.to_radians().sin(), cx: 760.0, cy: 145.0 },
        ];
        let mut matches = Vec::new();
        for (i, j) in [(0usize, 1usize), (1, 2), (0, 2)] {
            for gy in 0..12 {
                for gx in 0..40 {
                    let q = (gx as f64 * 25.0, gy as f64 * 25.0 + 10.0);
                    let (pi, pj) = (truth[i].invert(q), truth[j].invert(q));
                    if pi.0.abs() < w / 2.0 && pi.1.abs() < h / 2.0 && pj.0.abs() < w / 2.0 && pj.1.abs() < h / 2.0 {
                        matches.push(PointMatch { i, j, pi, pj, w: 1.0 });
                    }
                }
            }
        }
        // a few gross outliers
        for k in 0..5 {
            matches.push(PointMatch { i: 0, j: 1, pi: (150.0, k as f64 * 20.0), pj: (-100.0, k as f64 * 20.0 + 40.0), w: 1.0 });
        }
        let init: Vec<Pose> = truth.iter().map(|p| Pose { a: 0.0, b: 0.0, cx: p.cx + 3.0, cy: p.cy - 2.0 }).collect();
        let fit = solve_poses(&init, &matches, w);
        assert!(fit.rms_after < 0.2, "rms {} -> {}", fit.rms_before, fit.rms_after);
        for t in 0..3 {
            let rel = fit.poses[t].angle_deg() - fit.poses[0].angle_deg();
            let want = truth[t].angle_deg() - truth[0].angle_deg();
            assert!((rel - want).abs() < 0.02, "tile {t}: {rel:.3}° instead of {want:.3}°");
        }
    }
}
