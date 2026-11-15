//! Mosaic export: global alignment, brightness matching, multiband blending, writing.
//!
//! The blending follows the approach of Layr's stitcher (itself after OpenPano / Brown & Lowe):
//! every output pixel belongs to the tile it sits most centrally in, fine detail comes from that
//! tile alone (no ghosting), and coarser bands are mixed over wider strips around the seams
//! (Burt & Adelson). Brightness is matched between neighbours by least squares. The output is
//! processed in square blocks with a margin wide enough for the coarsest band, so memory stays
//! bounded whatever the mosaic's size.
//!
//! Unlike photos taken from different angles, our tiles only differ by a translation, which the
//! live tracker already estimated. Here that estimate is refined: phase correlation between
//! every pair of overlapping tiles gives relative offsets, and a least-squares solve removes the
//! drift accumulated along the scan.

use crate::session::{Session, TileMeta};
use crate::tracking::{gray_small, to_f32, PhaseCorrelator};
use anyhow::{bail, Context, Result};
use image::{imageops, GrayImage, RgbImage, RgbaImage};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

// ---------------------------------------------------------------- options / progress

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
}

impl Format {
    /// Format from a file extension (PNG when unknown).
    pub fn from_path(p: &Path) -> Format {
        match p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
            Some("jpg") | Some("jpeg") => Format::Jpeg,
            Some("tif") | Some("tiff") => Format::Tiff,
            _ => Format::Png,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Crop to the largest rectangle fully covered by tiles. Off by default: that rectangle can
    /// be much smaller than the board when the scan is ragged; otherwise the whole bounding box is
    /// kept, transparent where nothing was captured.
    pub crop: bool,
    /// Output scale (0.25..=1.0 of the capture resolution).
    pub scale: f32,
    /// Hard cap on output pixels; the scale is lowered to respect it.
    pub max_pixels: u64,
    /// Refine tile positions by phase correlation + least squares.
    pub refine: bool,
    /// Match brightness between neighbouring tiles.
    pub match_brightness: bool,
    pub jpeg_quality: u8,
    /// Reduction factor of the whole-tile registration pass (2 by default, 4 = faster; 1 was
    /// measured worse on real sessions).
    pub coarse_k: u32,
    /// Largest side (px) of the overlap crops used by the fine alignment. Larger is slower and was
    /// measured *worse* on real sessions (fewer reliable pairs): keep 512 unless experimenting.
    pub fine_side: u32,
    /// Memory budget for decoded tiles, in bytes (per pass).
    pub cache_bytes: usize,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            crop: false,
            scale: 1.0,
            max_pixels: 100_000_000,
            refine: true,
            match_brightness: true,
            jpeg_quality: 95,
            coarse_k: 2,
            fine_side: 512,
            cache_bytes: 1 << 30,
        }
    }
}

/// Progress shared with the UI. Cancel by calling [`Progress::cancel`].
pub struct Progress {
    stage: Mutex<String>,
    bits: AtomicU32,
    cancelled: AtomicBool,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Progress {
    pub fn new(wake: Option<Arc<dyn Fn() + Send + Sync>>) -> Arc<Self> {
        Arc::new(Self {
            stage: Mutex::new(String::new()),
            bits: AtomicU32::new(0f32.to_bits()),
            cancelled: AtomicBool::new(false),
            wake,
        })
    }
    pub fn set(&self, stage: &str, frac: f32) {
        if let Ok(mut s) = self.stage.lock() {
            if *s != stage {
                *s = stage.to_string();
            }
        }
        self.bits.store(frac.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        if let Some(w) = &self.wake {
            w();
        }
    }
    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
    }
    pub fn stage(&self) -> String {
        self.stage.lock().map(|s| s.clone()).unwrap_or_default()
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct ExportResult {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub tiles: usize,
    /// Overlapping pairs that gave a reliable measurement.
    pub pairs_used: usize,
    /// Largest displacement of a tile caused by the global alignment (capture-resolution px).
    pub max_correction: f64,
    /// Scale actually used (1.0 = capture resolution).
    pub scale: f32,
    /// Mean brightness-corrected difference (gray levels) between tiles where they overlap, after
    /// alignment: the lower, the better registered. `None` without registration data.
    pub overlap_error: Option<f64>,
}

// ---------------------------------------------------------------- tile loading

struct Cache {
    map: HashMap<usize, (Arc<RgbImage>, u64)>,
    bytes: usize,
    tick: u64,
}

/// The session's tiles, decoded on demand into a memory-bounded cache.
struct TileSet {
    dir: PathBuf,
    meta: Vec<TileMeta>,
    scale: f32,
    budget: usize,
    cache: Mutex<Cache>,
}

impl TileSet {
    fn new(dir: &Path, meta: Vec<TileMeta>, scale: f32, budget: usize) -> Self {
        Self {
            dir: dir.to_path_buf(),
            meta,
            scale,
            budget,
            cache: Mutex::new(Cache { map: HashMap::new(), bytes: 0, tick: 0 }),
        }
    }

    /// Size of tile `i` at this set's scale.
    fn size(&self, i: usize) -> (i64, i64) {
        let m = &self.meta[i];
        (
            ((m.width as f32 * self.scale).round() as i64).max(1),
            ((m.height as f32 * self.scale).round() as i64).max(1),
        )
    }

    fn load(&self, i: usize) -> Result<Arc<RgbImage>> {
        {
            let mut c = self.cache.lock().map_err(|_| anyhow::anyhow!("tile cache poisoned"))?;
            c.tick += 1;
            let t = c.tick;
            if let Some(e) = c.map.get_mut(&i) {
                e.1 = t;
                return Ok(e.0.clone());
            }
        }
        let path = self.dir.join(&self.meta[i].file);
        let mut img = image::open(&path).with_context(|| format!("reading {}", path.display()))?.to_rgb8();
        if self.scale < 1.0 {
            let (w, h) = self.size(i);
            img = imageops::resize(&img, w as u32, h as u32, imageops::FilterType::Triangle);
        }
        let img = Arc::new(img);
        let mut c = self.cache.lock().map_err(|_| anyhow::anyhow!("tile cache poisoned"))?;
        c.tick += 1;
        let t = c.tick;
        c.bytes += img.as_raw().len();
        c.map.insert(i, (img.clone(), t));
        while c.bytes > self.budget && c.map.len() > 1 {
            let oldest = c.map.iter().filter(|(k, _)| **k != i).min_by_key(|(_, v)| v.1).map(|(k, _)| *k);
            match oldest {
                Some(k) => {
                    if let Some((im, _)) = c.map.remove(&k) {
                        c.bytes -= im.as_raw().len();
                    }
                }
                None => break,
            }
        }
        Ok(img)
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).clamp(1, 6)
}

// ---------------------------------------------------------------- linear algebra

/// Conjugate gradient for a symmetric positive-definite operator.
fn cg(apply: &dyn Fn(&[f64], &mut [f64]), b: &[f64], x0: &[f64], iters: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = x0.to_vec();
    let mut ax = vec![0.0; n];
    apply(&x, &mut ax);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut p = r.clone();
    let mut rs = r.iter().map(|v| v * v).sum::<f64>();
    let b2 = b.iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let mut ap = vec![0.0; n];
    for _ in 0..iters {
        if rs < 1e-18 * b2 {
            break;
        }
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rs / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = r.iter().map(|v| v * v).sum::<f64>();
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }
    x
}

// ---------------------------------------------------------------- global alignment

/// Relative offset measured between two tiles: wanted `p_j - p_i`, with a weight.
#[derive(Debug, Clone, Copy)]
pub struct PairMeasure {
    pub i: usize,
    pub j: usize,
    pub t: (f64, f64),
    pub w: f64,
}

/// Least-squares positions: minimise `Σ w ((p_j - p_i) - t)² + prior Σ |p - p0|²`.
/// The weak prior keeps the solution near the initial estimate and fixes the global offset.
pub fn refine_positions(p0: &[(f64, f64)], pairs: &[PairMeasure], prior: f64) -> Vec<(f64, f64)> {
    let n = p0.len();
    let solve_axis = |axis: usize| -> Vec<f64> {
        let get = |v: (f64, f64)| if axis == 0 { v.0 } else { v.1 };
        let x0: Vec<f64> = p0.iter().map(|&p| get(p)).collect();
        let mut b: Vec<f64> = x0.iter().map(|v| prior * v).collect();
        for pm in pairs {
            let t = get(pm.t);
            b[pm.j] += pm.w * t;
            b[pm.i] -= pm.w * t;
        }
        let apply = |x: &[f64], y: &mut [f64]| {
            for i in 0..n {
                y[i] = prior * x[i];
            }
            for pm in pairs {
                let d = pm.w * (x[pm.i] - x[pm.j]);
                y[pm.i] += d;
                y[pm.j] -= d;
            }
        };
        cg(&apply, &b, &x0, 2000)
    };
    let (xs, ys) = (solve_axis(0), solve_axis(1));
    xs.into_iter().zip(ys).collect()
}

/// Mean brightness of an overlap, seen from tile `i` and from tile `j`.
#[derive(Debug, Clone, Copy)]
pub struct PairStat {
    pub i: usize,
    pub j: usize,
    pub mi: f64,
    pub mj: f64,
    /// Overlap area, in megapixels.
    pub n: f64,
}

/// A gain per tile so that neighbours look equally bright where they overlap
/// (Brown & Lowe's gain compensation).
pub fn compute_gains(n: usize, stats: &[PairStat]) -> Vec<f64> {
    const SIGMA_N2: f64 = 100.0; // (10 gray levels)²
    const INV_SIGMA_G2: f64 = 25.0; // (0.2)⁻²: a gentle pull toward unit gain
    let mut b = vec![0.0; n];
    for s in stats {
        b[s.i] += s.n * INV_SIGMA_G2;
        b[s.j] += s.n * INV_SIGMA_G2;
    }
    let apply = |x: &[f64], y: &mut [f64]| {
        y.iter_mut().for_each(|v| *v = 0.0);
        for s in stats {
            let cross = 2.0 * s.n * s.mi * s.mj / SIGMA_N2;
            y[s.i] += (2.0 * s.n * s.mi * s.mi / SIGMA_N2 + s.n * INV_SIGMA_G2) * x[s.i] - cross * x[s.j];
            y[s.j] += (2.0 * s.n * s.mj * s.mj / SIGMA_N2 + s.n * INV_SIGMA_G2) * x[s.j] - cross * x[s.i];
        }
    };
    let g = cg(&apply, &b, &vec![1.0; n], 500);
    g.into_iter().map(|v| v.clamp(0.7, 1.4)).collect()
}

/// Integer placement of a tile.
fn rounded(p: (f64, f64)) -> (i64, i64) {
    (p.0.round() as i64, p.1.round() as i64)
}

/// Measures the overlap of tiles `i` and `j`: a phase-correlation offset and brightness stats.
fn measure_pair(
    set: &TileSet,
    i: usize,
    j: usize,
    p: &[(f64, f64)],
    pcs: &mut HashMap<(usize, usize), PhaseCorrelator>,
    fine_side: f64,
) -> Result<(Option<PairMeasure>, Option<PairStat>)> {
    let (xi, yi) = rounded(p[i]);
    let (xj, yj) = rounded(p[j]);
    let (wi, hi) = (set.meta[i].width as i64, set.meta[i].height as i64);
    let (wj, hj) = (set.meta[j].width as i64, set.meta[j].height as i64);
    let (x0, x1) = (xi.max(xj), (xi + wi).min(xj + wj));
    let (y0, y1) = (yi.max(yj), (yi + hi).min(yj + hj));
    let (ow, oh) = (x1 - x0, y1 - y0);
    if ow < 48 || oh < 48 || (ow * oh) as f64 / ((wi * hi).min(wj * hj) as f64) < 0.08 {
        return Ok((None, None));
    }
    let (ti, tj) = (set.load(i)?, set.load(j)?);
    let a = imageops::crop_imm(&*ti, (x0 - xi) as u32, (y0 - yi) as u32, ow as u32, oh as u32).to_image();
    let b = imageops::crop_imm(&*tj, (x0 - xj) as u32, (y0 - yj) as u32, ow as u32, oh as u32).to_image();
    let k = (ow.max(oh) as f64 / fine_side).ceil().max(1.0) as u32;
    let (ga, gb): (GrayImage, GrayImage) = (gray_small(&a, k), gray_small(&b, k));
    let (fa, fb) = (to_f32(&ga), to_f32(&gb));
    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
    let stat = PairStat { i, j, mi: mean(&fa), mj: mean(&fb), n: (ow * oh) as f64 / 1e6 };

    let key = (ga.width() as usize, ga.height() as usize);
    let pc = pcs.entry(key).or_insert_with(|| PhaseCorrelator::new(key.0, key.1));
    let s = pc.correlate(&fa, &fb);
    if !s.is_reliable() {
        return Ok((None, Some(stat)));
    }
    // b(x) ≈ a(x - d) with d = e_i - e_j  ⇒  wanted p_j - p_i = (rounded p_j - p_i) - d
    let (dx, dy) = (s.dx * k as f64, s.dy * k as f64);
    // A tile pair cannot be off by more than a fraction of the field: reject wild measurements.
    let limit = 0.25 * (wi.min(hi).min(wj).min(hj)) as f64;
    if dx.abs() > limit || dy.abs() > limit {
        return Ok((None, Some(stat)));
    }
    let area = (ow * oh) as f64 / ((wi * hi).min(wj * hj) as f64);
    let w = (s.peak as f64).clamp(0.05, 1.0).powi(2) * area.sqrt();
    let t = ((xj - xi) as f64 - dx, (yj - yi) as f64 - dy);
    Ok((Some(PairMeasure { i, j, t, w }), Some(stat)))
}

/// Pairs of tiles whose rectangles overlap enough to be measured.
fn overlapping_pairs(meta: &[TileMeta], p: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..meta.len() {
        for j in i + 1..meta.len() {
            let (xi, yi) = rounded(p[i]);
            let (xj, yj) = rounded(p[j]);
            let ox = (xi + meta[i].width as i64).min(xj + meta[j].width as i64) - xi.max(xj);
            let oy = (yi + meta[i].height as i64).min(yj + meta[j].height as i64) - yi.max(yj);
            if ox >= 48 && oy >= 48 {
                out.push((i, j));
            }
        }
    }
    out
}

// ---------------------------------------------------------------- robust registration

/// Minimum median patch correlation for a coarse offset to be believed.
const COARSE_MIN_SCORE: f64 = 0.5;
/// Minimum number of textured patches that must agree.
const COARSE_MIN_PATCHES: usize = 6;
/// Minimum overlap (fraction of a tile) for a coarse offset to be believed.
const COARSE_MIN_OVERLAP: f64 = 0.10;
/// A coarse offset this far from the recorded layout (fraction of the tile) needs stronger proof.
const FAR_FROM_RECORDED: f64 = 0.15;
const PATCH: i64 = 24;
const PATCH_MIN_VAR: f64 = 16.0;

/// A tile reduced for the coarse registration.
struct Small {
    w: usize,
    h: usize,
    /// Reduction factor relative to the capture resolution.
    k: u32,
    px: Vec<f32>,
}

/// Pairs of tiles that may overlap according to the recorded positions, with a generous margin:
/// the recorded layout can be off by a large part of the field (e.g. at a row change).
fn near_pairs(meta: &[TileMeta], p: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..meta.len() {
        for j in i + 1..meta.len() {
            let (dx, dy) = ((p[i].0 - p[j].0).abs(), (p[i].1 - p[j].1).abs());
            let wmin = meta[i].width.min(meta[j].width) as f64;
            let hmin = meta[i].height.min(meta[j].height) as f64;
            if dx < 1.35 * wmin && dy < 1.35 * hmin {
                out.push((i, j));
            }
        }
    }
    out
}

/// How well `a` and `b` agree at the content shift `(dx, dy)` (`b(x) ≈ a(x - d)`): the *median*
/// normalised cross-correlation over textured patches of the overlap, the number of such patches
/// and the overlap as a fraction of the image. `None` when too little texture can be compared.
///
/// A single correlation over the whole overlap is easily fooled by content that looks the same
/// after any shift along one axis (the straight edge of a board against the table, a long trace):
/// those regions agree at a wrong offset while every real feature disagrees. Patches that carry
/// texture vote one by one, so a few uniform or edge-only areas cannot outvote the components.
fn patch_score(a: &Small, b: &Small, dx: i64, dy: i64) -> Option<(f64, usize, f64)> {
    let (w, h) = (a.w as i64, a.h as i64);
    let (x0, x1) = (dx.max(0), (w + dx).min(w));
    let (y0, y1) = (dy.max(0), (h + dy).min(h));
    if x1 - x0 < PATCH || y1 - y0 < PATCH {
        return None;
    }
    let frac = ((x1 - x0) * (y1 - y0)) as f64 / (w * h) as f64;
    let n = (PATCH * PATCH) as f64;
    let mut scores: Vec<f64> = Vec::new();
    let mut y = y0;
    while y + PATCH <= y1 {
        let mut x = x0;
        while x + PATCH <= x1 {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for py in 0..PATCH {
                let (ra, rb) = (((y + py - dy) * w) as usize, ((y + py) * w) as usize);
                for px in 0..PATCH {
                    let va = a.px[ra + (x + px - dx) as usize] as f64;
                    let vb = b.px[rb + (x + px) as usize] as f64;
                    sa += va;
                    sb += vb;
                    saa += va * va;
                    sbb += vb * vb;
                    sab += va * vb;
                }
            }
            let (va, vb) = ((saa - sa * sa / n) / n, (sbb - sb * sb / n) / n);
            if va > PATCH_MIN_VAR && vb > PATCH_MIN_VAR {
                scores.push(((sab - sa * sb / n) / n) / (va * vb).sqrt());
            }
            x += PATCH;
        }
        y += PATCH;
    }
    if scores.len() < 4 {
        return None;
    }
    scores.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    Some((scores[scores.len() / 2], scores.len(), frac))
}

/// Coarse measurement of a tile pair.
#[derive(Debug, Clone, Copy)]
struct Probe {
    /// Content shift in reduced pixels (`b(x) ≈ a(x - d)`).
    dx: f64,
    dy: f64,
    score: f64,
    patches: usize,
    frac: f64,
    /// Found by searching near the recorded layout rather than anywhere.
    near_recorded: bool,
}

/// Measures the offset of two whole tiles. `recorded` is the layout's `p_j - p_i` (capture px).
///
/// Candidates: the global phase-correlation peak (tried at both wraps of each axis, since phase
/// correlation is cyclic and shifts beyond half the image come out wrapped by one image size),
/// and the best peak near the recorded layout. Each is scored by [`patch_score`]. The candidate
/// near the recorded layout wins unless another is clearly better, since the live tracker is
/// right for most neighbours.
fn coarse_probe(a: &Small, b: &Small, pc: &mut PhaseCorrelator, recorded: (f64, f64)) -> Option<Probe> {
    if a.w != b.w || a.h != b.h {
        return None;
    }
    let k = a.k as f64;
    let alt = |d: f64, n: usize| if d >= 0.0 { d - n as f64 } else { d + n as f64 };
    let eval = |dx: f64, dy: f64, near: bool| -> Option<Probe> {
        let (score, patches, frac) = patch_score(a, b, dx.round() as i64, dy.round() as i64)?;
        Some(Probe { dx, dy, score, patches, frac, near_recorded: near })
    };
    let better = |best: &mut Option<Probe>, c: Option<Probe>| {
        if let Some(c) = c {
            if best.map_or(true, |b| c.score > b.score) {
                *best = Some(c);
            }
        }
    };

    let mut best: Option<Probe> = None;
    let s = pc.correlate(&a.px, &b.px);
    for &dx in &[s.dx, alt(s.dx, a.w)] {
        for &dy in &[s.dy, alt(s.dy, a.h)] {
            better(&mut best, eval(dx, dy, false));
        }
    }

    // Best peak near the recorded layout (expected content shift d0 = -recorded / k).
    let (d0x, d0y) = (-recorded.0 / k, -recorded.1 / k);
    let radius = (0.12 * a.w as f64) as usize;
    let near = pc.correlate_near(&a.px, &b.px, d0x, d0y, radius);
    let pick = |d: f64, d0: f64, n: usize| {
        // the reported peak is wrapped; keep the representative closest to the expectation
        [d, alt(d, n)].into_iter().min_by(|p, q| (p - d0).abs().partial_cmp(&(q - d0).abs()).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(d)
    };
    let hinted = eval(pick(near.dx, d0x, a.w), pick(near.dy, d0y, a.h), true);

    match (hinted, best) {
        (Some(h), Some(g)) if h.score >= COARSE_MIN_SCORE && h.score >= g.score - 0.15 => Some(h),
        (Some(h), Some(g)) => Some(if h.score >= g.score { h } else { g }),
        (Some(h), None) => Some(h),
        (None, g) => g,
    }
}

/// Decides whether a probe is trustworthy: `Some((p_j - p_i in capture px, weight))`.
fn coarse_pair(probe: &Probe, k: u32, recorded: (f64, f64), tile_min: f64) -> Option<((f64, f64), f64)> {
    if probe.score < COARSE_MIN_SCORE || probe.patches < COARSE_MIN_PATCHES || probe.frac < COARSE_MIN_OVERLAP {
        return None;
    }
    // b(x) ≈ a(x - d)  ⇒  p_j - p_i = -d  (whole tiles: no assumed placement)
    let t = (-probe.dx * k as f64, -probe.dy * k as f64);
    // Far from what the live tracker recorded: demand stronger evidence.
    let far = (t.0 - recorded.0).hypot(t.1 - recorded.1) > FAR_FROM_RECORDED * tile_min;
    if far && (probe.score < 0.7 || probe.patches < 8) {
        return None;
    }
    Some((t, probe.score * probe.score * probe.frac.sqrt()))
}

/// Mean absolute difference of `a` and `b` over their overlap at content shift `(dx, dy)`, after
/// removing each side's mean (so a brightness offset does not count): `(error, overlap pixels)`.
fn overlap_diff(a: &Small, b: &Small, dx: i64, dy: i64) -> Option<(f64, f64)> {
    let (w, h) = (a.w as i64, a.h as i64);
    let (x0, x1) = (dx.max(0), (w + dx).min(w));
    let (y0, y1) = (dy.max(0), (h + dy).min(h));
    if x1 - x0 < 16 || y1 - y0 < 16 {
        return None;
    }
    let n = ((x1 - x0) * (y1 - y0)) as f64;
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for y in y0..y1 {
        for x in x0..x1 {
            sa += a.px[((y - dy) * w + x - dx) as usize] as f64;
            sb += b.px[(y * w + x) as usize] as f64;
        }
    }
    let off = (sb - sa) / n;
    let mut err = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            err += (b.px[(y * w + x) as usize] as f64 - a.px[((y - dy) * w + x - dx) as usize] as f64 - off).abs();
        }
    }
    Some((err / n, n))
}

/// How well the final layout registers the tiles: overlap-weighted mean of [`overlap_diff`] over
/// every pair that overlaps by at least 10 % of a tile.
fn overlap_error(smalls: &[Option<Small>], positions: &[(f64, f64)]) -> Option<f64> {
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for i in 0..smalls.len() {
        for j in i + 1..smalls.len() {
            let (Some(a), Some(b)) = (&smalls[i], &smalls[j]) else { continue };
            if a.w != b.w || a.h != b.h {
                continue;
            }
            let k = a.k as f64;
            // b(x) ≈ a(x - d), d = -(p_j - p_i) / k
            let d = (-(positions[j].0 - positions[i].0) / k, -(positions[j].1 - positions[i].1) / k);
            if let Some((e, n)) = overlap_diff(a, b, d.0.round() as i64, d.1.round() as i64) {
                if n >= 0.10 * (a.w * a.h) as f64 {
                    sum += e * n;
                    weight += n;
                }
            }
        }
    }
    (weight > 0.0).then(|| sum / weight)
}

/// Solves for positions, then drops the measurements that disagree with the solution
/// (more than 3× the median residual, and at least `cut_min` px) and solves again.
fn solve_robust(
    start: &[(f64, f64)],
    measures: Vec<PairMeasure>,
    cut_min: f64,
) -> (Vec<(f64, f64)>, Vec<PairMeasure>) {
    if measures.is_empty() {
        return (start.to_vec(), measures);
    }
    let total_w: f64 = measures.iter().map(|m| m.w).sum();
    let prior = 1e-3 * total_w / start.len() as f64;
    let pos = refine_positions(start, &measures, prior);
    let residual = |pos: &[(f64, f64)], m: &PairMeasure| {
        (pos[m.j].0 - pos[m.i].0 - m.t.0).hypot(pos[m.j].1 - pos[m.i].1 - m.t.1)
    };
    let mut res: Vec<f64> = measures.iter().map(|m| residual(&pos, m)).collect();
    res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cut = (3.0 * res[res.len() / 2]).max(cut_min);
    let kept: Vec<PairMeasure> = measures.iter().filter(|m| residual(&pos, m) <= cut).copied().collect();
    let (pos, kept) = if kept.len() < measures.len() && !kept.is_empty() {
        (refine_positions(start, &kept, prior), kept)
    } else {
        (pos, measures)
    };
    if std::env::var_os("PCBMAPPER_DEBUG").is_some() {
        let mut r: Vec<(f64, usize, usize)> = kept.iter().map(|m| (residual(&pos, m), m.i, m.j)).collect();
        r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mean = r.iter().map(|x| x.0).sum::<f64>() / r.len().max(1) as f64;
        eprintln!("solve: {} pairs, mean residual {:.2} px, worst {:?}", r.len(), mean, &r[..r.len().min(6)]);
    }
    (pos, kept)
}

// ---------------------------------------------------------------- crop

/// Largest axis-aligned rectangle of `true` cells: `(x, y, w, h)` in cells.
pub fn largest_rect(grid: &[bool], gw: usize, gh: usize) -> (usize, usize, usize, usize) {
    let mut heights = vec![0usize; gw];
    let mut best = (0usize, 0usize, 0usize, 0usize);
    let mut best_area = 0usize;
    for y in 0..gh {
        for x in 0..gw {
            heights[x] = if grid[y * gw + x] { heights[x] + 1 } else { 0 };
        }
        // Largest rectangle in the histogram (monotonic stack)
        let mut stack: Vec<usize> = Vec::new();
        for x in 0..=gw {
            let cur = if x < gw { heights[x] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] <= cur {
                    break;
                }
                stack.pop();
                let h = heights[top];
                let left = stack.last().map_or(0, |&l| l + 1);
                let w = x - left;
                if h * w > best_area {
                    best_area = h * w;
                    best = (left, y + 1 - h, w, h);
                }
            }
            stack.push(x.min(gw.saturating_sub(1)));
            if x == gw {
                break;
            }
        }
    }
    best
}

/// Largest rectangle fully covered by the tiles (canvas px): `(x, y, w, h)`.
fn covered_rect(rects: &[(i64, i64, i64, i64)], cw: i64, ch: i64) -> (i64, i64, i64, i64) {
    let cell = ((cw.max(ch) as f64 / 1500.0).ceil() as i64).max(4);
    let (gw, gh) = (((cw + cell - 1) / cell) as usize, ((ch + cell - 1) / cell) as usize);
    let mut grid = vec![false; gw * gh];
    for &(x, y, w, h) in rects {
        // cells lying entirely inside this tile
        let (cx0, cy0) = ((x + cell - 1).div_euclid(cell).max(0), (y + cell - 1).div_euclid(cell).max(0));
        let (cx1, cy1) = (((x + w).div_euclid(cell)).min(gw as i64), ((y + h).div_euclid(cell)).min(gh as i64));
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                grid[cy as usize * gw + cx as usize] = true;
            }
        }
    }
    let (gx, gy, rw, rh) = largest_rect(&grid, gw, gh);
    if rw == 0 || rh == 0 {
        return (0, 0, cw, ch);
    }
    (gx as i64 * cell, gy as i64 * cell, (rw as i64 * cell).min(cw), (rh as i64 * cell).min(ch))
}

// ---------------------------------------------------------------- multiband blending

const BANDS: usize = 5;
const MARGIN: usize = 128;
const CORE: usize = 512;
const NONE: u16 = u16::MAX;
const K5: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

#[derive(Clone)]
struct Plane {
    w: usize,
    h: usize,
    d: Vec<f32>,
}

impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Self { w, h, d: vec![0.0; w * h] }
    }

    /// Blurs with the 5-tap binomial kernel and keeps every other pixel.
    fn down(&self) -> Plane {
        let (w, h) = (self.w, self.h);
        let (nw, nh) = ((w + 1) / 2, (h + 1) / 2);
        let mut tmp = Plane::new(nw, h);
        for y in 0..h {
            let row = &self.d[y * w..(y + 1) * w];
            for x in 0..nw {
                let mut s = 0.0;
                for (t, k) in K5.iter().enumerate() {
                    let sx = (2 * x + t).saturating_sub(2).min(w - 1);
                    s += k * row[sx];
                }
                tmp.d[y * nw + x] = s;
            }
        }
        let mut out = Plane::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let mut s = 0.0;
                for (t, k) in K5.iter().enumerate() {
                    let sy = (2 * y + t).saturating_sub(2).min(h - 1);
                    s += k * tmp.d[sy * nw + x];
                }
                out.d[y * nw + x] = s;
            }
        }
        out
    }

    /// Resizes to `w × h` by bilinear interpolation.
    fn up(&self, w: usize, h: usize) -> Plane {
        let mut out = Plane::new(w, h);
        let (fx, fy) = (self.w as f32 / w as f32, self.h as f32 / h as f32);
        for y in 0..h {
            let sy = ((y as f32 + 0.5) * fy - 0.5).max(0.0);
            let y0 = (sy as usize).min(self.h - 1);
            let y1 = (y0 + 1).min(self.h - 1);
            let ty = sy - y0 as f32;
            for x in 0..w {
                let sx = ((x as f32 + 0.5) * fx - 0.5).max(0.0);
                let x0 = (sx as usize).min(self.w - 1);
                let x1 = (x0 + 1).min(self.w - 1);
                let tx = sx - x0 as f32;
                let a = self.d[y0 * self.w + x0] * (1.0 - tx) + self.d[y0 * self.w + x1] * tx;
                let b = self.d[y1 * self.w + x0] * (1.0 - tx) + self.d[y1 * self.w + x1] * tx;
                out.d[y * w + x] = a * (1.0 - ty) + b * ty;
            }
        }
        out
    }
}

fn gaussian_pyramid(base: Plane) -> Vec<Plane> {
    let mut v = vec![base];
    for _ in 0..BANDS {
        let next = v.last().map(|p| p.down()).unwrap_or_else(|| Plane::new(1, 1));
        v.push(next);
    }
    v
}

/// A tile placed on the canvas (canvas px, at the export scale).
struct Placed {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

/// Blends the canvas block whose extended region starts at `(ex, ey)`, `ew × eh` px wide.
/// Returns the three colour planes and which pixels are covered by at least one tile.
fn blend_block(
    set: &TileSet,
    placed: &[Placed],
    gains: &[f32],
    contrib: &[usize],
    (ex, ey, ew, eh): (i64, i64, usize, usize),
) -> Result<([Plane; 3], Vec<bool>)> {
    // Owner of each pixel: the tile it sits most centrally in.
    let mut best = vec![0f32; ew * eh];
    let mut owner = vec![NONE; ew * eh];
    for (ci, &ti) in contrib.iter().enumerate() {
        let pl = &placed[ti];
        for y in 0..eh {
            let gy = ey + y as i64;
            if gy < pl.y || gy >= pl.y + pl.h {
                continue;
            }
            let v = ((gy - pl.y) as f32 + 0.5) / pl.h as f32;
            let wy = 0.5 - (v - 0.5).abs();
            for x in 0..ew {
                let gx = ex + x as i64;
                if gx < pl.x || gx >= pl.x + pl.w {
                    continue;
                }
                let u = ((gx - pl.x) as f32 + 0.5) / pl.w as f32;
                let wgt = wy * (0.5 - (u - 0.5).abs());
                if wgt > best[y * ew + x] {
                    best[y * ew + x] = wgt;
                    owner[y * ew + x] = ci as u16;
                }
            }
        }
    }
    let covered: Vec<bool> = owner.iter().map(|&o| o != NONE).collect();

    // Pyramids of every contributor: its colour (edge-replicated beyond the tile) and its mask.
    let mut masks: Vec<Vec<Plane>> = Vec::new();
    let mut colours: Vec<[Vec<Plane>; 3]> = Vec::new();
    for (ci, &ti) in contrib.iter().enumerate() {
        let img = set.load(ti)?;
        let pl = &placed[ti];
        let g = gains[ti];
        let mut m = Plane::new(ew, eh);
        for (i, &o) in owner.iter().enumerate() {
            if o as usize == ci {
                m.d[i] = 1.0;
            }
        }
        let mut ch = [Plane::new(ew, eh), Plane::new(ew, eh), Plane::new(ew, eh)];
        let raw = img.as_raw();
        let stride = img.width() as usize * 3;
        for y in 0..eh {
            let sy = (ey + y as i64 - pl.y).clamp(0, pl.h - 1) as usize;
            for x in 0..ew {
                let sx = (ex + x as i64 - pl.x).clamp(0, pl.w - 1) as usize;
                let o = sy * stride + sx * 3;
                for c in 0..3 {
                    ch[c].d[y * ew + x] = raw[o + c] as f32 * g;
                }
            }
        }
        masks.push(gaussian_pyramid(m));
        let [r, g2, b] = ch;
        colours.push([gaussian_pyramid(r), gaussian_pyramid(g2), gaussian_pyramid(b)]);
    }

    // Blend band by band: Σ mask_k · band_k / Σ mask_k.
    let mut result: [Vec<Plane>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for k in 0..=BANDS {
        let (lw, lh) = (masks[0][k].w, masks[0][k].h);
        let mut den = Plane::new(lw, lh);
        for m in &masks {
            for (d, v) in den.d.iter_mut().zip(&m[k].d) {
                *d += v;
            }
        }
        for c in 0..3 {
            let mut num = Plane::new(lw, lh);
            for (ci, m) in masks.iter().enumerate() {
                let band = if k < BANDS {
                    let up = colours[ci][c][k + 1].up(lw, lh);
                    let g = &colours[ci][c][k];
                    let mut l = Plane::new(lw, lh);
                    for i in 0..l.d.len() {
                        l.d[i] = g.d[i] - up.d[i];
                    }
                    l
                } else {
                    colours[ci][c][k].clone()
                };
                for i in 0..num.d.len() {
                    num.d[i] += m[k].d[i] * band.d[i];
                }
            }
            for i in 0..num.d.len() {
                num.d[i] = if den.d[i] > 1e-4 { num.d[i] / den.d[i] } else { 0.0 };
            }
            result[c].push(num);
        }
    }
    // Collapse the pyramid.
    let mut out: Vec<Plane> = Vec::new();
    for c in 0..3 {
        let mut acc = result[c][BANDS].clone();
        for k in (0..BANDS).rev() {
            let band = &result[c][k];
            let mut up = acc.up(band.w, band.h);
            for i in 0..up.d.len() {
                up.d[i] += band.d[i];
            }
            acc = up;
        }
        out.push(acc);
    }
    let [a, b, c] = <[Plane; 3]>::try_from(out).map_err(|_| anyhow::anyhow!("internal: plane count"))?;
    Ok(([a, b, c], covered))
}

// ---------------------------------------------------------------- export

fn save(img: &RgbaImage, path: &Path, format: Format, quality: u8) -> Result<()> {
    let opaque = img.pixels().all(|p| p.0[3] == 255);
    let rgb = || image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    match format {
        Format::Png | Format::Tiff => {
            let f = if format == Format::Png { image::ImageFormat::Png } else { image::ImageFormat::Tiff };
            if opaque {
                rgb().save_with_format(path, f)?;
            } else {
                img.save_with_format(path, f)?;
            }
        }
        Format::Jpeg => {
            let file = std::fs::File::create(path)?;
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), quality);
            enc.encode_image(&rgb())?;
        }
    }
    Ok(())
}

/// Exports the session in `session_dir` to `out`.
pub fn export_session(
    session_dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    progress: &Progress,
) -> Result<ExportResult> {
    let session = Session::open(session_dir).context("opening the session")?;
    let meta = session.meta.tiles.clone();
    let n = meta.len();
    if n == 0 {
        bail!("the session has no tiles yet");
    }
    let p0: Vec<(f64, f64)> = meta.iter().map(|t| (t.x, t.y)).collect();

    // ---- 1) coarse registration: whole tiles, independent of the recorded layout
    let mut start = p0.clone();
    let mut coarse_pairs = 0;
    let mut kept_smalls: Vec<Option<Small>> = Vec::new();
    if opts.refine && n > 1 {
        progress.set("Registering tiles", 0.0);
        let coarse_k = opts.coarse_k.clamp(1, 8);
        let coarse_set = TileSet::new(session_dir, meta.clone(), 1.0, opts.cache_bytes);
        let smalls: Mutex<Vec<Option<Small>>> = Mutex::new((0..n).map(|_| None).collect());
        let next = AtomicUsize::new(0);
        let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads().min(n) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n || progress.is_cancelled() {
                        break;
                    }
                    match coarse_set.load(i) {
                        Ok(img) => {
                            let g = gray_small(&img, coarse_k);
                            let small = Small { w: g.width() as usize, h: g.height() as usize, k: coarse_k, px: to_f32(&g) };
                            if let Ok(mut v) = smalls.lock() {
                                v[i] = Some(small);
                            }
                        }
                        Err(e) => {
                            if let Ok(mut f) = failure.lock() {
                                f.get_or_insert(e);
                            }
                            break;
                        }
                    }
                    progress.set("Registering tiles", 0.05 * (i + 1) as f32 / n as f32);
                });
            }
        });
        if let Some(e) = failure.lock().ok().and_then(|mut f| f.take()) {
            return Err(e);
        }
        drop(coarse_set);
        let smalls: Vec<Option<Small>> = smalls.into_inner().unwrap_or_default();
        let debug = std::env::var_os("PCBMAPPER_DEBUG").is_some();
        let cand = near_pairs(&meta, &p0);
        let found: Mutex<Vec<PairMeasure>> = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads().min(cand.len().max(1)) {
                s.spawn(|| {
                    let mut pcs: HashMap<(usize, usize), PhaseCorrelator> = HashMap::new();
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= cand.len() || progress.is_cancelled() {
                            break;
                        }
                        let (i, j) = cand[idx];
                        if let (Some(a), Some(b)) = (&smalls[i], &smalls[j]) {
                            let pc = pcs.entry((a.w, a.h)).or_insert_with(|| PhaseCorrelator::new_tapered(a.w, a.h, 0.15));
                            let recorded = (p0[j].0 - p0[i].0, p0[j].1 - p0[i].1);
                            let tile_min = meta[i].width.min(meta[i].height) as f64;
                            let probe = coarse_probe(a, b, pc, recorded);
                            let verdict = probe.as_ref().and_then(|pr| coarse_pair(pr, a.k, recorded, tile_min));
                            if debug {
                                if let Some(pr) = &probe {
                                    eprintln!(
                                        "coarse {i:>3}-{j:>3}: shift {:>7.1},{:>7.1} px  score {:.2} ({} patches)  overlap {:.0} %  {}  recorded {:>7.1},{:>7.1}  -> {}",
                                        -pr.dx * a.k as f64,
                                        -pr.dy * a.k as f64,
                                        pr.score,
                                        pr.patches,
                                        pr.frac * 100.0,
                                        if pr.near_recorded { "near" } else { "free" },
                                        recorded.0,
                                        recorded.1,
                                        if verdict.is_some() { "accepted" } else { "rejected" }
                                    );
                                }
                            }
                            if let Some((t, w)) = verdict {
                                if let Ok(mut f) = found.lock() {
                                    f.push(PairMeasure { i, j, t, w });
                                }
                            }
                        }
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        progress.set("Registering tiles", 0.05 + 0.15 * d as f32 / cand.len() as f32);
                    }
                });
            }
        });
        if progress.is_cancelled() {
            bail!("export cancelled");
        }
        let found = found.into_inner().unwrap_or_default();
        let found_n = found.len();
        kept_smalls = smalls;
        let (pos, kept) = solve_robust(&p0, found, 12.0);
        coarse_pairs = kept.len();
        if debug {
            eprintln!("coarse: {} candidates, {} accepted, {} kept", cand.len(), found_n, coarse_pairs);
        }
        start = pos;
    }

    // ---- 2) fine alignment on the corrected layout: phase correlation of the overlaps + brightness
    progress.set("Measuring overlaps", 0.2);
    let measure_set = TileSet::new(session_dir, meta.clone(), 1.0, opts.cache_bytes);
    let pairs = overlapping_pairs(&meta, &start);
    let measures: Mutex<Vec<PairMeasure>> = Mutex::new(Vec::new());
    let stats: Mutex<Vec<PairStat>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    if opts.refine || opts.match_brightness {
        std::thread::scope(|s| {
            for _ in 0..threads().min(pairs.len().max(1)) {
                s.spawn(|| {
                    let mut pcs = HashMap::new();
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= pairs.len() || progress.is_cancelled() {
                            break;
                        }
                        let (i, j) = pairs[idx];
                        match measure_pair(&measure_set, i, j, &start, &mut pcs, opts.fine_side.max(128) as f64) {
                            Ok((m, st)) => {
                                if let (Some(m), Ok(mut g)) = (m, measures.lock()) {
                                    g.push(m);
                                }
                                if let (Some(st), Ok(mut g)) = (st, stats.lock()) {
                                    g.push(st);
                                }
                            }
                            Err(e) => {
                                if let Ok(mut f) = failure.lock() {
                                    f.get_or_insert(e);
                                }
                                break;
                            }
                        }
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        progress.set("Measuring overlaps", 0.2 + 0.15 * d as f32 / pairs.len() as f32);
                    }
                });
            }
        });
    }
    if let Some(e) = failure.lock().ok().and_then(|mut f| f.take()) {
        return Err(e);
    }
    if progress.is_cancelled() {
        bail!("export cancelled");
    }
    drop(measure_set);
    let measures = measures.into_inner().unwrap_or_default();
    let stats = stats.into_inner().unwrap_or_default();

    progress.set("Aligning tiles", 0.36);
    let mut positions = start.clone();
    let mut pairs_used = coarse_pairs;
    if opts.refine && !measures.is_empty() {
        let (pos, kept) = solve_robust(&start, measures, 4.0);
        positions = pos;
        pairs_used = kept.len();
    }
    let overlap_error = if kept_smalls.is_empty() { None } else { overlap_error(&kept_smalls, &positions) };
    drop(kept_smalls);
    let max_correction = positions
        .iter()
        .zip(&p0)
        .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
        .fold(0.0, f64::max);

    // ---- 3) brightness
    progress.set("Matching brightness", 0.38);
    let gains: Vec<f32> = if opts.match_brightness && !stats.is_empty() {
        compute_gains(n, &stats).into_iter().map(|g| g as f32).collect()
    } else {
        vec![1.0; n]
    };

    // ---- 4) canvas geometry (and scale, to respect the pixel cap)
    let min_x = positions.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let min_y = positions.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let max_x = positions.iter().zip(&meta).map(|(p, m)| p.0 + m.width as f64).fold(f64::MIN, f64::max);
    let max_y = positions.iter().zip(&meta).map(|(p, m)| p.1 + m.height as f64).fold(f64::MIN, f64::max);
    let (full_w, full_h) = ((max_x - min_x).ceil().max(1.0), (max_y - min_y).ceil().max(1.0));
    let mut scale = opts.scale.clamp(0.05, 1.0) as f64;
    let cap = (opts.max_pixels as f64 / (full_w * full_h)).sqrt();
    if cap < scale {
        scale = cap;
    }
    let scale = scale as f32;
    let blend_set = TileSet::new(session_dir, meta.clone(), scale, opts.cache_bytes);
    let placed: Vec<Placed> = (0..n)
        .map(|i| {
            let (w, h) = blend_set.size(i);
            Placed {
                x: ((positions[i].0 - min_x) * scale as f64).round() as i64,
                y: ((positions[i].1 - min_y) * scale as f64).round() as i64,
                w,
                h,
            }
        })
        .collect();
    let (cw, ch) = (
        placed.iter().map(|p| p.x + p.w).max().unwrap_or(1),
        placed.iter().map(|p| p.y + p.h).max().unwrap_or(1),
    );
    let (rx, ry, rw, rh) = if opts.crop {
        let rects: Vec<_> = placed.iter().map(|p| (p.x, p.y, p.w, p.h)).collect();
        covered_rect(&rects, cw, ch)
    } else {
        (0, 0, cw, ch)
    };
    if rw < 1 || rh < 1 {
        bail!("empty export area");
    }

    // ---- 5) blend block by block
    let mut blocks = Vec::new();
    let (mut by, mut ny) = (ry, 0);
    while by < ry + rh {
        let bh = (CORE as i64).min(ry + rh - by);
        let (mut bx, mut nx) = (rx, 0);
        while bx < rx + rw {
            let bw = (CORE as i64).min(rx + rw - bx);
            blocks.push((bx, by, bw, bh, nx, ny));
            bx += bw;
            nx += 1;
        }
        by += bh;
        ny += 1;
    }
    let mut canvas = RgbaImage::new(rw as u32, rh as u32);
    let gains_ref = &gains;
    let (placed_ref, set_ref) = (&placed, &blend_set);
    let next = AtomicUsize::new(0);
    let (tx, rx_chan) = mpsc::channel::<Result<(i64, i64, i64, i64, Vec<u8>)>>();
    std::thread::scope(|s| {
        for _ in 0..threads().min(blocks.len().max(1)) {
            let tx = tx.clone();
            let (blocks, next) = (&blocks, &next);
            s.spawn(move || loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= blocks.len() || progress.is_cancelled() {
                    break;
                }
                let (bx, by, bw, bh, _, _) = blocks[idx];
                let ext = (bx - MARGIN as i64, by - MARGIN as i64, bw as usize + 2 * MARGIN, bh as usize + 2 * MARGIN);
                let contrib: Vec<usize> = (0..placed_ref.len())
                    .filter(|&i| {
                        let p = &placed_ref[i];
                        p.x < ext.0 + ext.2 as i64 && p.x + p.w > ext.0 && p.y < ext.1 + ext.3 as i64 && p.y + p.h > ext.1
                    })
                    .collect();
                let mut rgba = vec![0u8; (bw * bh * 4) as usize];
                let res: Result<()> = if contrib.is_empty() {
                    Ok(())
                } else {
                    blend_block(set_ref, placed_ref, gains_ref, &contrib, ext).map(|(planes, covered)| {
                        for y in 0..bh as usize {
                            for x in 0..bw as usize {
                                let ei = (y + MARGIN) * ext.2 + x + MARGIN;
                                if covered[ei] {
                                    let o = (y * bw as usize + x) * 4;
                                    for c in 0..3 {
                                        rgba[o + c] = planes[c].d[ei].round().clamp(0.0, 255.0) as u8;
                                    }
                                    rgba[o + 3] = 255;
                                }
                            }
                        }
                    })
                };
                let _ = tx.send(res.map(|_| (bx, by, bw, bh, rgba)));
            });
        }
        drop(tx);
        let mut received = 0usize;
        for msg in rx_chan.iter() {
            match msg {
                Ok((bx, by, bw, bh, rgba)) => {
                    for y in 0..bh as usize {
                        let dst = (((by - ry) as usize + y) * rw as usize + (bx - rx) as usize) * 4;
                        let src = y * bw as usize * 4;
                        canvas.as_mut()[dst..dst + bw as usize * 4].copy_from_slice(&rgba[src..src + bw as usize * 4]);
                    }
                }
                Err(e) => {
                    if let Ok(mut f) = failure.lock() {
                        f.get_or_insert(e);
                    }
                    progress.cancel();
                }
            }
            received += 1;
            progress.set("Blending", 0.4 + 0.55 * received as f32 / blocks.len() as f32);
        }
    });
    if let Some(e) = failure.lock().ok().and_then(|mut f| f.take()) {
        return Err(e);
    }
    if progress.is_cancelled() {
        bail!("export cancelled");
    }

    // ---- 6) write
    progress.set("Writing the file", 0.96);
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    save(&canvas, out, Format::from_path(out), opts.jpeg_quality)
        .with_context(|| format!("writing {}", out.display()))?;
    progress.set("Done", 1.0);
    Ok(ExportResult {
        path: out.to_path_buf(),
        width: rw as u32,
        height: rh as u32,
        tiles: n,
        pairs_used,
        max_correction,
        scale,
        overlap_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::synthetic_board;
    use image::Rgb;

    #[test]
    fn refine_positions_removes_accumulated_drift() {
        // 4 x 3 grid, true spacing 100 px; the initial estimates drift along each row.
        let truth: Vec<(f64, f64)> = (0..12).map(|k| ((k % 4) as f64 * 100.0, (k / 4) as f64 * 70.0)).collect();
        let init: Vec<(f64, f64)> = truth
            .iter()
            .enumerate()
            .map(|(k, &(x, y))| (x + (k % 4) as f64 * 3.0 - (k / 4) as f64 * 2.0, y + (k % 4) as f64 * 1.5))
            .collect();
        let mut pairs = Vec::new();
        for k in 0..12 {
            if k % 4 < 3 {
                pairs.push((k, k + 1));
            }
            if k / 4 < 2 {
                pairs.push((k, k + 4));
            }
        }
        let measures: Vec<PairMeasure> = pairs
            .iter()
            .map(|&(i, j)| PairMeasure { i, j, t: (truth[j].0 - truth[i].0, truth[j].1 - truth[i].1), w: 1.0 })
            .collect();
        let r = refine_positions(&init, &measures, 1e-3);
        for k in 0..12 {
            let (ex, ey) = (r[k].0 - r[0].0 - (truth[k].0 - truth[0].0), r[k].1 - r[0].1 - (truth[k].1 - truth[0].1));
            assert!(ex.abs() < 0.2 && ey.abs() < 0.2, "tile {k}: {ex:.2},{ey:.2}");
        }
    }

    #[test]
    fn gains_equalise_overlaps() {
        // tile 1 is 10 % darker, tile 2 is 8 % brighter than tile 0
        let base = 120.0;
        let level = [1.0, 0.9, 1.08];
        let stats = vec![
            PairStat { i: 0, j: 1, mi: base * level[0], mj: base * level[1], n: 1.0 },
            PairStat { i: 1, j: 2, mi: base * level[1], mj: base * level[2], n: 1.0 },
            PairStat { i: 0, j: 2, mi: base * level[0], mj: base * level[2], n: 1.0 },
        ];
        let g = compute_gains(3, &stats);
        let seen: Vec<f64> = (0..3).map(|i| g[i] * level[i]).collect();
        let spread = seen.iter().cloned().fold(f64::MIN, f64::max) - seen.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 0.02, "{seen:?}");
    }

    #[test]
    fn largest_rect_in_an_l_shape() {
        // 6 x 4 grid, the top-right 3 x 2 block is empty
        let mut grid = vec![true; 24];
        for y in 0..2 {
            for x in 3..6 {
                grid[y * 6 + x] = false;
            }
        }
        let (x, y, w, h) = largest_rect(&grid, 6, 4);
        assert_eq!(w * h, 12); // the full-height left part (3 x 4) or the full-width bottom (6 x 2)
        assert!(grid[y * 6 + x] && grid[(y + h - 1) * 6 + x + w - 1]);
    }

    #[test]
    fn covered_rect_ignores_ragged_edges() {
        // two rows of tiles, the second one shifted right: the clean rectangle is their common part
        let rects = vec![(0, 0, 400, 300), (350, 0, 400, 300), (100, 250, 400, 300), (450, 250, 400, 300)];
        let (x, y, w, h) = covered_rect(&rects, 850, 550);
        assert!(x >= 98 && x <= 110 && y <= 8, "{x},{y}");
        assert!(w >= 640 && w <= 652 && h >= 540, "{w}x{h}");
    }

    /// Writes a session whose tiles come from a known board, with drifting position estimates and
    /// per-tile brightness differences, exports it and compares the result with the board.
    #[test]
    fn export_reconstructs_the_board_despite_drift_and_brightness() {
        let board = synthetic_board(2265, 1274);
        let (tw, th) = (960u32, 540u32);
        let root = std::env::temp_dir().join(format!("pcbmapper-export-{}", std::process::id()));
        let mut session = Session::create(&root).unwrap();
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
        };
        let (mut drift_x, mut drift_y) = (0.0, 0.0);
        let mut k = 0;
        for &ty in &[0u32, 367, 734] {
            for &tx in &[0u32, 652, 1305] {
                let mut tile = imageops::crop_imm(&board, tx, ty, tw, th).to_image();
                let gain = [1.0, 0.92, 1.07, 0.96, 1.05, 0.98, 1.04, 0.94, 1.0][k];
                for p in tile.pixels_mut() {
                    *p = Rgb([
                        (p.0[0] as f64 * gain).min(255.0) as u8,
                        (p.0[1] as f64 * gain).min(255.0) as u8,
                        (p.0[2] as f64 * gain).min(255.0) as u8,
                    ]);
                }
                drift_x += rnd() * 5.0;
                drift_y += rnd() * 5.0;
                session.add_tile(&tile, (tx as f64 + drift_x, ty as f64 + drift_y)).unwrap();
                k += 1;
            }
        }
        let out = root.join("mosaic.png");
        let progress = Progress::new(None);
        let r = export_session(session.dir(), &out, &ExportOptions::default(), &progress).unwrap();
        eprintln!("export: {}x{}, {} pairs, max correction {:.1} px", r.width, r.height, r.pairs_used, r.max_correction);
        assert!(r.pairs_used >= 10, "only {} pairs measured", r.pairs_used);
        let mosaic = image::open(&out).unwrap().to_rgb8();
        assert!(mosaic.width() >= 2200 && mosaic.height() >= 1230, "{}x{}", mosaic.width(), mosaic.height());

        // Compare with the board over the central area, searching the small global offset.
        let mut best = f64::MAX;
        for oy in 0..=14i64 {
            for ox in 0..=14i64 {
                let (mut sum, mut cnt) = (0.0f64, 0u32);
                for y in (150..mosaic.height().min(1000)).step_by(5) {
                    for x in (150..mosaic.width().min(2000)).step_by(5) {
                        let (bx, by) = (x as i64 + ox - 7, y as i64 + oy - 7);
                        if bx < 0 || by < 0 || bx >= 2265 || by >= 1274 {
                            continue;
                        }
                        let a = mosaic.get_pixel(x, y).0;
                        let b = board.get_pixel(bx as u32, by as u32).0;
                        sum += (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>() / 3.0;
                        cnt += 1;
                    }
                }
                best = best.min(sum / cnt.max(1) as f64);
            }
        }
        eprintln!("mean abs difference to the board: {best:.2} gray levels");
        assert!(best < 4.0, "the mosaic differs from the board by {best:.2} levels");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn single_tile_is_reproduced_exactly() {
        let board = synthetic_board(700, 500);
        let root = std::env::temp_dir().join(format!("pcbmapper-export1-{}", std::process::id()));
        let mut session = Session::create(&root).unwrap();
        let tile = imageops::crop_imm(&board, 20, 10, 640, 400).to_image();
        session.add_tile(&tile, (0.0, 0.0)).unwrap();
        let out = root.join("one.png");
        export_session(session.dir(), &out, &ExportOptions::default(), &Progress::new(None)).unwrap();
        let mosaic = image::open(&out).unwrap().to_rgb8();
        assert_eq!(mosaic.dimensions(), tile.dimensions());
        let worst = mosaic.pixels().zip(tile.pixels()).map(|(a, b)| (0..3).map(|c| (a.0[c] as i32 - b.0[c] as i32).abs()).max().unwrap_or(0)).max().unwrap_or(0);
        assert!(worst <= 2, "worst difference {worst}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn export_scale_and_cap_shrink_the_output() {
        let board = synthetic_board(900, 600);
        let root = std::env::temp_dir().join(format!("pcbmapper-export2-{}", std::process::id()));
        let mut session = Session::create(&root).unwrap();
        session.add_tile(&imageops::crop_imm(&board, 0, 0, 600, 400).to_image(), (0.0, 0.0)).unwrap();
        session.add_tile(&imageops::crop_imm(&board, 300, 200, 600, 400).to_image(), (300.0, 200.0)).unwrap();
        let out = root.join("small.jpg");
        let opts = ExportOptions { scale: 0.5, crop: false, ..Default::default() };
        let r = export_session(session.dir(), &out, &opts, &Progress::new(None)).unwrap();
        assert!((r.width as i32 - 450).abs() <= 3 && (r.height as i32 - 300).abs() <= 3, "{}x{}", r.width, r.height);
        let capped = ExportOptions { max_pixels: 90_000, crop: false, ..Default::default() };
        let r2 = export_session(session.dir(), &root.join("cap.png"), &capped, &Progress::new(None)).unwrap();
        assert!(r2.width as u64 * r2.height as u64 <= 95_000, "{}x{}", r2.width, r2.height);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Reproduces a real failure: the recorded layout is wrong by 300 px at the row change and the
    /// tiles overlap by only 32 % (shift = 68 % of the width, beyond the half image where phase
    /// correlation wraps). The export must still register everything from the images alone.
    #[test]
    fn export_recovers_from_a_badly_recorded_row_change() {
        let board = synthetic_board(2265, 1274);
        let (tw, th) = (960u32, 540u32);
        let root = std::env::temp_dir().join(format!("pcbmapper-export3-{}", std::process::id()));
        let mut session = Session::create(&root).unwrap();
        for &(ty, wrong_y) in &[(0u32, 0.0f64), (367, 0.0), (734, 0.0)] {
            // second and third rows are recorded 300 px too low
            let recorded_dy = if ty > 0 { 300.0 } else { wrong_y };
            for &tx in &[0u32, 652, 1305] {
                let tile = imageops::crop_imm(&board, tx, ty, tw, th).to_image();
                session.add_tile(&tile, (tx as f64, ty as f64 + recorded_dy)).unwrap();
            }
        }
        let out = root.join("mosaic.png");
        let r = export_session(session.dir(), &out, &ExportOptions::default(), &Progress::new(None)).unwrap();
        eprintln!("export: {}x{}, {} pairs, max correction {:.1} px", r.width, r.height, r.pairs_used, r.max_correction);
        let mosaic = image::open(&out).unwrap().to_rgb8();
        assert!(mosaic.width() >= 2200 && mosaic.height() >= 1230 && mosaic.height() <= 1330, "{}x{}", mosaic.width(), mosaic.height());
        let mut best = f64::MAX;
        for oy in 0..=14i64 {
            for ox in 0..=14i64 {
                let (mut sum, mut cnt) = (0.0f64, 0u32);
                for y in (100..mosaic.height().min(1150)).step_by(5) {
                    for x in (100..mosaic.width().min(2100)).step_by(5) {
                        let (bx, by) = (x as i64 + ox - 7, y as i64 + oy - 7);
                        if bx < 0 || by < 0 || bx >= 2265 || by >= 1274 {
                            continue;
                        }
                        let a = mosaic.get_pixel(x, y).0;
                        let b = board.get_pixel(bx as u32, by as u32).0;
                        sum += (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>() / 3.0;
                        cnt += 1;
                    }
                }
                best = best.min(sum / cnt.max(1) as f64);
            }
        }
        eprintln!("mean abs difference to the board: {best:.2} gray levels");
        assert!(best < 4.0, "the mosaic differs from the board by {best:.2} levels");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The top of a board is a straight edge against a flat table: it matches after *any* shift
    /// along x. A global correlation is fooled; the patch vote is not.
    #[test]
    fn patch_score_rejects_a_shift_that_only_the_board_edge_supports() {
        let scene = |x: i64, y: i64| -> f32 {
            if y < 50 {
                40.0 // flat table
            } else if ((x * 37 + y * 91).rem_euclid(97)) < 7 {
                255.0 // components
            } else if y < 56 {
                230.0 // bright board edge, identical at every x
            } else {
                120.0 + ((x * 13 + y * 7).rem_euclid(5)) as f32
            }
        };
        let make = |ox: i64| Small {
            w: 240,
            h: 135,
            k: 4,
            px: (0..135).flat_map(|y| (0..240).map(move |x| (x as i64, y as i64))).map(|(x, y)| scene(x + ox, y)).collect(),
        };
        let (a, b) = (make(0), make(100)); // b(x) = a(x + 100)  ⇒  content shift d = -100
        let (good, good_patches, _) = patch_score(&a, &b, -100, 0).expect("enough texture");
        let (bad, _, _) = patch_score(&a, &b, -40, 0).expect("enough texture");
        assert!(good > 0.9 && good_patches >= 6, "true shift scored {good:.2} on {good_patches} patches");
        assert!(bad < 0.5, "a wrong shift scored {bad:.2}");
    }
}
