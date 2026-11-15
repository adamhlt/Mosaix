//! Motion tracking by phase correlation (FFT).
//!
//! Convention: `correlate(a, b)` returns `(dx, dy)` such that `b(x, y) ≈ a(x - dx, y - dy)`,
//! i.e. how far the image **content** moved between `a` and `b`.

use image::{GrayImage, RgbImage};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Result of a phase correlation.
#[derive(Debug, Clone, Copy)]
pub struct Shift {
    /// Content displacement in pixels (sub-pixel), in the scale of the supplied images.
    pub dx: f64,
    pub dy: f64,
    /// Peak height (≈ 1 for a perfect shift, → 0 with no correlation).
    pub peak: f32,
    /// Peak-to-sidelobe ratio (PSR): low with blur, glare or repetitive patterns.
    pub psr: f32,
}

impl Shift {
    /// Quality deemed good enough to trust the displacement.
    pub fn is_reliable(&self) -> bool {
        self.peak > 0.04 && self.psr > 6.0
    }
}

/// Phase correlator for fixed-size images (FFT plans are reused).
pub struct PhaseCorrelator {
    w: usize,
    h: usize,
    fft_row: Arc<dyn Fft<f32>>,
    fft_col: Arc<dyn Fft<f32>>,
    ifft_row: Arc<dyn Fft<f32>>,
    ifft_col: Arc<dyn Fft<f32>>,
    hann_x: Vec<f32>,
    hann_y: Vec<f32>,
    buf_a: Vec<Complex<f32>>,
    buf_b: Vec<Complex<f32>>,
    tmp: Vec<Complex<f32>>,
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos())
        .collect()
}

impl PhaseCorrelator {
    pub fn new(w: usize, h: usize) -> Self {
        let mut p = FftPlanner::<f32>::new();
        Self {
            w,
            h,
            fft_row: p.plan_fft_forward(w),
            fft_col: p.plan_fft_forward(h),
            ifft_row: p.plan_fft_inverse(w),
            ifft_col: p.plan_fft_inverse(h),
            hann_x: hann(w),
            hann_y: hann(h),
            buf_a: vec![Complex::new(0.0, 0.0); w * h],
            buf_b: vec![Complex::new(0.0, 0.0); w * h],
            tmp: vec![Complex::new(0.0, 0.0); w * h],
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Sub-pixel position (offset in `[-1, 1]` around `p`) of the maximum of the
    /// correlation along one axis, from the cross-power spectrum stored in `buf_b`.
    /// `horizontal`: X axis (the peak is at `(p, q)`), otherwise Y axis (the peak is at `(q, p)`).
    fn refine_axis(&self, p: usize, q: usize, horizontal: bool) -> f64 {
        use std::f64::consts::TAU;
        let (w, h) = (self.w, self.h);
        let (n_main, n_other) = if horizontal { (w, h) } else { (h, w) };
        let signed = |k: usize, n: usize| -> f64 {
            if k > n / 2 { k as f64 - n as f64 } else { k as f64 }
        };
        // Projection onto the other axis, evaluated at its integer position q.
        let mut prof = vec![Complex::<f64>::new(0.0, 0.0); n_main];
        for (km, pm) in prof.iter_mut().enumerate() {
            let mut acc = Complex::<f64>::new(0.0, 0.0);
            for ko in 0..n_other {
                let v = if horizontal { self.buf_b[ko * w + km] } else { self.buf_b[km * w + ko] };
                let ph = TAU * signed(ko, n_other) * q as f64 / n_other as f64;
                let (s, c) = ph.sin_cos();
                acc += Complex::new(v.re as f64, v.im as f64) * Complex::new(c, s);
            }
            *pm = acc;
        }
        let eval = |u: f64| -> f64 {
            let pos = p as f64 + u;
            let mut acc = 0.0;
            for (km, c) in prof.iter().enumerate() {
                let ph = TAU * signed(km, n_main) * pos / n_main as f64;
                let (s, co) = ph.sin_cos();
                acc += c.re * co - c.im * s;
            }
            acc
        };
        let (mut best_u, mut best_v) = (0.0, f64::MIN);
        for i in -20..=20 {
            let u = i as f64 * 0.05;
            let v = eval(u);
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
        let c = best_u;
        for i in -10..=10 {
            let u = c + i as f64 * 0.005;
            let v = eval(u);
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
        best_u
    }

    /// In-place 2D FFT: rows, transpose, columns, transpose.
    fn fft2(
        w: usize,
        h: usize,
        row: &Arc<dyn Fft<f32>>,
        col: &Arc<dyn Fft<f32>>,
        data: &mut [Complex<f32>],
        tmp: &mut [Complex<f32>],
    ) {
        for r in data.chunks_exact_mut(w) {
            row.process(r);
        }
        transpose(data, tmp, w, h); // tmp: h columns of length… (h x w)
        for c in tmp.chunks_exact_mut(h) {
            col.process(c);
        }
        transpose(tmp, data, h, w);
    }

    /// Loads an image (f32 grayscale, mean removed) with a Hann window.
    fn load(w: usize, h: usize, hx: &[f32], hy: &[f32], src: &[f32], dst: &mut [Complex<f32>]) {
        let mean = src.iter().sum::<f32>() / src.len() as f32;
        for y in 0..h {
            for x in 0..w {
                let v = (src[y * w + x] - mean) * hx[x] * hy[y];
                dst[y * w + x] = Complex::new(v, 0.0);
            }
        }
    }

    /// Correlates two images of the correlator's size (`w*h` values, row by row).
    pub fn correlate(&mut self, a: &[f32], b: &[f32]) -> Shift {
        let (w, h) = (self.w, self.h);
        assert_eq!(a.len(), w * h);
        assert_eq!(b.len(), w * h);
        Self::load(w, h, &self.hann_x, &self.hann_y, a, &mut self.buf_a);
        Self::load(w, h, &self.hann_x, &self.hann_y, b, &mut self.buf_b);
        Self::fft2(w, h, &self.fft_row, &self.fft_col, &mut self.buf_a, &mut self.tmp);
        Self::fft2(w, h, &self.fft_row, &self.fft_col, &mut self.buf_b, &mut self.tmp);

        // Normalised cross-power spectrum: conj(A)·B / |conj(A)·B|, weighted by a
        // Gaussian low-pass (high frequencies, dominated by noise, bias the sub-pixel estimate).
        const SIGMA: f32 = 0.12; // in cycles/pixel
        let fx2: Vec<f32> = (0..w)
            .map(|k| {
                let f = if k > w / 2 { k as f32 - w as f32 } else { k as f32 } / w as f32;
                f * f
            })
            .collect();
        let fy2: Vec<f32> = (0..h)
            .map(|k| {
                let f = if k > h / 2 { k as f32 - h as f32 } else { k as f32 } / h as f32;
                f * f
            })
            .collect();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let c = self.buf_a[i].conj() * self.buf_b[i];
                let n = c.norm();
                let g = (-(fx2[x] + fy2[y]) / (2.0 * SIGMA * SIGMA)).exp();
                self.buf_a[i] = if n > 1e-9 { c * (g / n) } else { Complex::new(0.0, 0.0) };
            }
        }
        // Copy of the cross-power spectrum (used for sub-pixel refinement), then inverse FFT.
        self.buf_b.copy_from_slice(&self.buf_a);
        Self::fft2(w, h, &self.ifft_row, &self.ifft_col, &mut self.buf_a, &mut self.tmp);

        let norm = 1.0 / (w * h) as f32;
        // Main peak
        let (mut best, mut px, mut py) = (f32::MIN, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                let v = self.buf_a[y * w + x].re * norm;
                if v > best {
                    best = v;
                    px = x;
                    py = y;
                }
            }
        }

        // PSR: (peak - mean) / standard deviation of the background, outside the 11x11 neighbourhood of the peak.
        let (mut sum, mut sum2, mut n) = (0.0f64, 0.0f64, 0usize);
        for y in 0..h {
            for x in 0..w {
                let dx = circ_dist(x, px, w);
                let dy = circ_dist(y, py, h);
                if dx > 5 || dy > 5 {
                    let v = (self.buf_a[y * w + x].re * norm) as f64;
                    sum += v;
                    sum2 += v * v;
                    n += 1;
                }
            }
        }
        let mean = sum / n.max(1) as f64;
        let var = (sum2 / n.max(1) as f64 - mean * mean).max(1e-18);
        let psr = ((best as f64 - mean) / var.sqrt()) as f32;

        // Sub-pixel refinement: the inverse correlation is evaluated at fractional positions
        // directly from the cross-power spectrum (non-uniform DFT, separable, along the row
        // and column of the peak), then the continuous maximum is searched. A 3-point
        // parabolic fit is biased because the peak is very narrow.
        let fx = self.refine_axis(px, py, true);
        let fy = self.refine_axis(py, px, false);

        let wrap = |p: usize, n: usize| -> f64 {
            if p > n / 2 {
                p as f64 - n as f64
            } else {
                p as f64
            }
        };
        Shift { dx: wrap(px, w) + fx, dy: wrap(py, h) + fy, peak: best, psr }
    }
}

fn circ_dist(a: usize, b: usize, n: usize) -> usize {
    let d = a.abs_diff(b);
    d.min(n - d)
}

/// Transposes `src` (w columns × h rows) into `dst` (h columns × w rows).
fn transpose(src: &[Complex<f32>], dst: &mut [Complex<f32>], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            dst[x * h + y] = src[y * w + x];
        }
    }
}

/// Downscales an RGB image to grayscale by averaging `k × k` blocks (fast, no costly
/// filter). Takes ≈ 3 ms for 1920×1080 with `k = 4`.
pub fn gray_small(img: &RgbImage, k: u32) -> GrayImage {
    let k = k.max(1);
    let (w, h) = (img.width() / k, img.height() / k);
    let src = img.as_raw();
    let stride = img.width() as usize * 3;
    let mut out = GrayImage::new(w, h);
    let dst = out.as_mut();
    let area = (k * k) as u32;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for dy in 0..k as usize {
                let row = (y * k as usize + dy) * stride + x * k as usize * 3;
                for dx in 0..k as usize {
                    let i = row + dx * 3;
                    r += src[i] as u32;
                    g += src[i + 1] as u32;
                    b += src[i + 2] as u32;
                }
            }
            // Rec.601 luminance (weights out of 256)
            dst[y * w as usize + x] = ((77 * r + 150 * g + 29 * b) / (256 * area)) as u8;
        }
    }
    out
}

/// `f32` version of a grayscale image.
pub fn to_f32(img: &GrayImage) -> Vec<f32> {
    img.as_raw().iter().map(|&v| v as f32).collect()
}

/// Downscales an RGB image by averaging `k × k` blocks (mosaic thumbnails).
pub fn rgb_downscale(img: &RgbImage, k: u32) -> RgbImage {
    let k = k.max(1);
    let (w, h) = ((img.width() / k).max(1), (img.height() / k).max(1));
    let src = img.as_raw();
    let stride = img.width() as usize * 3;
    let mut out = RgbImage::new(w, h);
    let area = k * k;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let mut acc = [0u32; 3];
            for dy in 0..k as usize {
                let row = (y * k as usize + dy) * stride + x * k as usize * 3;
                for dx in 0..k as usize {
                    for c in 0..3 {
                        acc[c] += src[row + dx * 3 + c] as u32;
                    }
                }
            }
            out.put_pixel(x as u32, y as u32, image::Rgb([(acc[0] / area) as u8, (acc[1] / area) as u8, (acc[2] / area) as u8]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic synthetic image: sum of Gaussians + details, evaluable at
    /// floating-point coordinates (allows exact sub-pixel shifts).
    fn texture(x: f64, y: f64) -> f32 {
        let mut s = 0.0f64;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..90 {
            let (cx, cy) = (rnd() * 400.0 - 50.0, rnd() * 300.0 - 50.0);
            let sg = 2.0 + rnd() * 6.0;
            let amp = rnd() * 2.0 - 1.0;
            let d2 = (x - cx).powi(2) + (y - cy).powi(2);
            s += amp * (-d2 / (2.0 * sg * sg)).exp();
        }
        (128.0 + 60.0 * s) as f32
    }

    fn render(w: usize, h: usize, dx: f64, dy: f64) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                v.push(texture(x as f64 - dx, y as f64 - dy));
            }
        }
        v
    }

    #[test]
    fn integer_shift() {
        let (w, h) = (256, 192);
        let mut pc = PhaseCorrelator::new(w, h);
        let a = render(w, h, 0.0, 0.0);
        for &(dx, dy) in &[(0.0, 0.0), (7.0, -3.0), (-20.0, 11.0), (35.0, 28.0)] {
            let b = render(w, h, dx, dy);
            let s = pc.correlate(&a, &b);
            assert!((s.dx - dx).abs() < 0.5 && (s.dy - dy).abs() < 0.5, "{dx},{dy} -> {s:?}");
            assert!(s.is_reliable(), "{s:?}");
        }
    }

    #[test]
    fn subpixel_shift() {
        let (w, h) = (256, 192);
        let mut pc = PhaseCorrelator::new(w, h);
        let a = render(w, h, 0.0, 0.0);
        for &(dx, dy) in &[(0.5, 0.0), (3.3, -2.7), (-10.25, 6.75), (1.8, 1.2)] {
            let b = render(w, h, dx, dy);
            let s = pc.correlate(&a, &b);
            assert!((s.dx - dx).abs() < 0.2 && (s.dy - dy).abs() < 0.2, "{dx},{dy} -> {s:?}");
        }
    }

    #[test]
    fn uncorrelated_is_unreliable() {
        let (w, h) = (256, 192);
        let mut pc = PhaseCorrelator::new(w, h);
        let a = render(w, h, 0.0, 0.0);
        // Completely different content: pseudo-random noise
        let mut seed = 99u32;
        let b: Vec<f32> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as f32
            })
            .collect();
        let s = pc.correlate(&a, &b);
        assert!(!s.is_reliable(), "{s:?}");
    }

    #[test]
    fn speed_budget() {
        // 480x270 (1920x1080 downscaled to 0.25) must stay well under 10 ms in release;
        // in tests (opt-level 2) we only check a reasonable order of magnitude.
        let (w, h) = (480, 270);
        let mut pc = PhaseCorrelator::new(w, h);
        let a = render(w, h, 0.0, 0.0);
        let b = render(w, h, 5.0, 5.0);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            pc.correlate(&a, &b);
        }
        let ms = t.elapsed().as_secs_f64() * 100.0;
        eprintln!("480x270 correlation: {ms:.2} ms");
        assert!(ms < 40.0);
    }
}
