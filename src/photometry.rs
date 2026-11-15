//! Fields of the camera frame (vignetting, uneven lighting, a tilted focal plane) and per-tile
//! factors (exposure, white balance, motion blur), fitted from tile overlaps.
//!
//! Tiles of a scan differ by a translation only, so a board point seen in two tiles sits at two
//! different places of the frame. Comparing the two observations separates what belongs to the
//! tile from what belongs to the position in the frame:
//!
//! `y = ln obs_i − ln obs_j = (t_i + F(u_i, v_i)) − (t_j + F(u_j, v_j)) + noise`
//!
//! with one value `t` per tile and a smooth polynomial `F` of the frame coordinates
//! (`u, v ∈ [-1, 1]`). The linear part of `F` cannot be told apart from a brightness ramp across
//! the board absorbed by the `t`s (add `a·x_i` to every `t_i` and `−a·u` to `F`): a prior pulling
//! the `t`s together lifts the ambiguity and attributes what is tied to the frame to the frame,
//! which is what a fixed microscope light does. Outliers (misregistered or moving content,
//! clipped pixels) are down-weighted by iteratively reweighted least squares (Huber).

/// Number of polynomial terms of a frame field (no constant: it is carried by the tiles).
pub const NF: usize = 12;

/// Polynomial terms of a frame field at `(u, v)`.
pub fn terms(u: f64, v: f64) -> [f64; NF] {
    let (u2, v2) = (u * u, v * v);
    [u, v, u2, v2, u * v, u2 * u, v2 * v, u2 * v, u * v2, u2 * u2, v2 * v2, u2 * v2]
}

/// One board point seen in tiles `i` and `j`: `d = terms(u_i, v_i) − terms(u_j, v_j)`.
#[derive(Debug, Clone, Copy)]
pub struct FieldSample {
    pub i: u32,
    pub j: u32,
    pub d: [f32; NF],
    pub y: f32,
    /// Confidence of the observation (1 = full).
    pub weight: f32,
}

impl FieldSample {
    pub fn new(i: usize, j: usize, (ui, vi): (f64, f64), (uj, vj): (f64, f64), y: f64) -> Self {
        let (a, b) = (terms(ui, vi), terms(uj, vj));
        let mut d = [0f32; NF];
        for k in 0..NF {
            d[k] = (a[k] - b[k]) as f32;
        }
        Self { i: i as u32, j: j as u32, d, y: y as f32, weight: 1.0 }
    }
}

/// A fitted model: `ln obs = tile[i] + frame(u, v) + ln truth`.
#[derive(Debug, Clone)]
pub struct FrameField {
    pub tile: Vec<f64>,
    pub coef: [f64; NF],
    /// RMS of the observations before the fit and robust spread of the residuals after it.
    pub rms_before: f64,
    pub sigma_after: f64,
    pub samples: usize,
}

impl FrameField {
    /// The neutral model: nothing to correct.
    pub fn flat(n: usize) -> Self {
        Self { tile: vec![0.0; n], coef: [0.0; NF], rms_before: 0.0, sigma_after: 0.0, samples: 0 }
    }

    pub fn frame(&self, u: f64, v: f64) -> f64 {
        terms(u, v).iter().zip(&self.coef).map(|(t, c)| t * c).sum()
    }

    /// Mean of the frame field over the whole frame.
    pub fn frame_mean(&self) -> f64 {
        let (nx, ny) = (32, 18);
        let mut s = 0.0;
        for y in 0..ny {
            for x in 0..nx {
                s += self.frame(2.0 * (x as f64 + 0.5) / nx as f64 - 1.0, 2.0 * (y as f64 + 0.5) / ny as f64 - 1.0);
            }
        }
        s / (nx * ny) as f64
    }
}

/// Fits a [`FrameField`] to `samples`. `tile_prior` is the strength of the pull of every tile term
/// toward zero, as a fraction of the average number of observations per tile; `frame_prior` the
/// pull of every polynomial coefficient toward zero, as a fraction of all observations. The latter
/// keeps the field flat in directions the overlaps do not constrain (a single row of tiles says
/// little about the vertical profile) at the cost of shrinking well-measured terms by about
/// `frame_prior / mean(d²)`.
pub fn fit(n: usize, samples: &[FieldSample], tile_prior: f64, frame_prior: f64) -> FrameField {
    if n == 0 || samples.len() < 8 {
        return FrameField::flat(n);
    }
    let m = n + NF;
    let rms_before = (samples.iter().map(|s| (s.y as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    let base: Vec<f64> = samples.iter().map(|s| (s.weight as f64).max(0.0)).collect();
    let mut w = base.clone();
    let mut x = vec![0.0f64; m];
    let mut sigma = rms_before.max(1e-6);
    for round in 0..6 {
        let total: f64 = w.iter().sum();
        let lt = tile_prior * total / n as f64;
        let lc = frame_prior * total;
        // right-hand side Aᵀ W y
        let mut b = vec![0.0f64; m];
        let mut diag = vec![0.0f64; m];
        for (s, &ws) in samples.iter().zip(&w) {
            let (i, j) = (s.i as usize, s.j as usize);
            let wy = ws * s.y as f64;
            b[i] += wy;
            b[j] -= wy;
            diag[i] += ws;
            diag[j] += ws;
            for k in 0..NF {
                let d = s.d[k] as f64;
                b[n + k] += wy * d;
                diag[n + k] += ws * d * d;
            }
        }
        for v in &mut diag[..n] {
            *v += lt;
        }
        for v in &mut diag[n..] {
            *v += lc;
        }
        let apply = |x: &[f64], out: &mut [f64]| {
            for (o, xi) in out[..n].iter_mut().zip(&x[..n]) {
                *o = lt * xi;
            }
            for (o, xi) in out[n..].iter_mut().zip(&x[n..]) {
                *o = lc * xi;
            }
            for (s, &ws) in samples.iter().zip(&w) {
                let (i, j) = (s.i as usize, s.j as usize);
                let mut p = x[i] - x[j];
                for k in 0..NF {
                    p += s.d[k] as f64 * x[n + k];
                }
                let wp = ws * p;
                out[i] += wp;
                out[j] -= wp;
                for k in 0..NF {
                    out[n + k] += wp * s.d[k] as f64;
                }
            }
        };
        x = pcg(&apply, &b, &diag, &x, 400);
        // robust reweighting
        let mut res: Vec<f64> = samples.iter().map(|s| residual(s, &x, n).abs()).collect();
        let mut sorted = res.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sigma = (1.4826 * sorted[sorted.len() / 2]).max(1e-4);
        if round == 5 {
            break;
        }
        let cut = 1.5 * sigma;
        for ((wi, r), b) in w.iter_mut().zip(res.iter_mut()).zip(&base) {
            *wi = b * if *r <= cut { 1.0 } else { cut / *r };
        }
    }
    let mut coef = [0.0; NF];
    coef.copy_from_slice(&x[n..]);
    FrameField { tile: x[..n].to_vec(), coef, rms_before, sigma_after: sigma, samples: samples.len() }
}

fn residual(s: &FieldSample, x: &[f64], n: usize) -> f64 {
    let mut p = x[s.i as usize] - x[s.j as usize];
    for k in 0..NF {
        p += s.d[k] as f64 * x[n + k];
    }
    s.y as f64 - p
}

/// Jacobi-preconditioned conjugate gradient.
pub(crate) fn pcg(apply: &dyn Fn(&[f64], &mut [f64]), b: &[f64], diag: &[f64], x0: &[f64], iters: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = x0.to_vec();
    let mut ax = vec![0.0; n];
    apply(&x, &mut ax);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let inv: Vec<f64> = diag.iter().map(|d| if *d > 1e-30 { 1.0 / d } else { 1.0 }).collect();
    let mut z: Vec<f64> = r.iter().zip(&inv).map(|(r, i)| r * i).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let b2 = b.iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let mut ap = vec![0.0; n];
    for _ in 0..iters {
        if r.iter().map(|v| v * v).sum::<f64>() < 1e-20 * b2 {
            break;
        }
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        for i in 0..n {
            z[i] = r[i] * inv[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3 × 4 serpentine scan of a random board under a lopsided light, with per-tile exposure
    /// changes: the fit must recover the light's shape (up to a constant) and the exposures.
    #[test]
    fn recovers_vignetting_and_exposure() {
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let board: Vec<f64> = (0..400 * 300).map(|_| 3.0 + rnd()).collect(); // ln reflectance
        let light = |u: f64, v: f64| -0.35 * u - 0.25 * u * u - 0.15 * v * v + 0.1 * u * v;
        let (tw, th) = (160usize, 120usize);
        let origins: Vec<(usize, usize)> = (0..3).flat_map(|r| (0..4).map(move |c| (c * 75 + r * 3, r * 80 + c * 2))).collect();
        let expo: Vec<f64> = origins.iter().map(|_| 0.2 * (rnd() - 0.5)).collect();
        let obs = |t: usize, x: usize, y: usize| {
            let (ox, oy) = origins[t];
            let (u, v) = (2.0 * (x as f64 + 0.5) / tw as f64 - 1.0, 2.0 * (y as f64 + 0.5) / th as f64 - 1.0);
            board[(oy + y) * 400 + ox + x] + expo[t] + light(u, v)
        };
        let mut samples = Vec::new();
        for i in 0..origins.len() {
            for j in i + 1..origins.len() {
                let (dx, dy) = (origins[j].0 as i64 - origins[i].0 as i64, origins[j].1 as i64 - origins[i].1 as i64);
                for y in (0..th as i64).step_by(4) {
                    for x in (0..tw as i64).step_by(4) {
                        let (xj, yj) = (x - dx, y - dy);
                        if xj < 0 || yj < 0 || xj >= tw as i64 || yj >= th as i64 {
                            continue;
                        }
                        let uv = |x: i64, y: i64| (2.0 * (x as f64 + 0.5) / tw as f64 - 1.0, 2.0 * (y as f64 + 0.5) / th as f64 - 1.0);
                        let y_obs = obs(i, x as usize, y as usize) - obs(j, xj as usize, yj as usize);
                        samples.push(FieldSample::new(i, j, uv(x, y), uv(xj, yj), y_obs));
                    }
                }
            }
        }
        let f = fit(origins.len(), &samples, 0.05, 1e-4);
        assert!(f.sigma_after < 0.01, "residual {}", f.sigma_after);
        // The light's shape, up to an affine ramp: random exposures may correlate with the scan
        // position, and a ramp of exposures is indistinguishable from a linear term of the light.
        let e = |u: f64, v: f64| f.frame(u, v) - light(u, v);
        let (c, a, b) = (e(0.0, 0.0), (e(1.0, 0.0) - e(-1.0, 0.0)) / 2.0, (e(0.0, 1.0) - e(0.0, -1.0)) / 2.0);
        assert!(a.abs() < 0.1 && b.abs() < 0.1, "ramp {a:.3},{b:.3}");
        for &(u, v) in &[(-0.9, -0.8), (0.9, 0.0), (0.5, 0.7), (-0.3, 0.9), (0.8, -0.9)] {
            let d = e(u, v) - c - a * u - b * v;
            assert!(d.abs() < 0.02, "at {u},{v}: off by {d:.3}");
        }
        // without exposure changes, the prior has nothing to trade: the light comes out exactly
        let flat: Vec<FieldSample> =
            samples.iter().map(|s| FieldSample { y: s.y - (expo[s.i as usize] - expo[s.j as usize]) as f32, ..*s }).collect();
        let f = fit(origins.len(), &flat, 0.05, 1e-4);
        let c = f.frame(0.0, 0.0) - light(0.0, 0.0);
        for &(u, v) in &[(-0.9, -0.8), (0.9, 0.0), (0.5, 0.7)] {
            let d = f.frame(u, v) - light(u, v) - c;
            assert!(d.abs() < 0.02, "at {u},{v}: off by {d:.3}");
        }
        assert!(f.tile.iter().all(|t| t.abs() < 0.01), "{:?}", f.tile);
    }

    #[test]
    fn too_few_samples_is_neutral() {
        let f = fit(3, &[], 0.1, 1e-3);
        assert_eq!(f.tile, vec![0.0; 3]);
        assert_eq!(f.frame(0.5, 0.5), 0.0);
    }
}
