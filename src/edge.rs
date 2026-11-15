//! Automatic board-edge detection.
//!
//! The board lies on a table. When the view passes the board edge, a band of table appears on the
//! side the camera is heading to. Real captures show that the table is *not* flat (wood grain,
//! light falling off to 0.6× on one side, a tilted focal plane), so a "flat area" test misses it.
//! Instead the detector looks for a band that:
//!
//! 1. touches the leading border and covers (almost) all of it, line after line;
//! 2. is made of low-contrast blocks of one colour (chromaticity + brightness), which wood grain,
//!    a plastic case or a sheet of paper all are;
//! 3. looks like the table already seen, or else like nothing seen in the middle of the captured
//!    tiles (the *board model*): a plain solder-mask area is uniform too, but it also shows up
//!    inside the board.
//!
//! The table is learnt from the tiles themselves: a captured tile with a strongly novel band on
//! one of its sides (typically the first tile, taken in a corner of the board) teaches what the
//! table looks like, and that look is kept out of the board model. Gradients are divided by the
//! local brightness, colours are compared as chromaticities and brightness is corrected by a light
//! fall-off profile estimated from the tiles; blur only softens the board, it does not make it
//! table-coloured.

use image::{GrayImage, RgbImage};

#[derive(Debug, Clone)]
pub struct EdgeParams {
    /// Block side, in analysis-image pixels.
    pub block: u32,
    /// A block is "plain" when its mean gradient / (brightness + 4) is below this.
    pub flat_grad: f32,
    /// Two blocks have the same colour when their chromaticities differ by less than this.
    pub chroma_tol: f32,
    /// … and their brightness ratio is below this (generous: light is uneven across the field).
    pub lum_ratio: f32,
    /// Width of the outermost strip that defines the band's colour (fraction of the field).
    pub outer: f32,
    /// Share of plain blocks needed in that outermost strip.
    pub min_outer_plain: f32,
    /// A line (perpendicular to the heading) belongs to the band when this share of it matches.
    pub line_frac: f32,
    /// Band depth (fraction of the field) from which it counts as the board edge.
    pub min_depth: f32,
    /// Brightness ratio below which a band matches the board model or a known table.
    pub match_lum_ratio: f32,
    /// A band that does not look like a known table may match at most this share of the plain
    /// blocks of the board model.
    pub max_novelty: f32,
    /// A band that looks like a known table may match at most this share.
    pub max_novelty_known: f32,
    /// A band on a captured tile this novel, covering this share of each line, is learnt as the
    /// table (a table band runs along the whole border; a label or a connector does not).
    pub learn_novelty: f32,
    pub learn_line_frac: f32,
    /// Darker bands are ignored (lens cap, dead frame).
    pub min_lum: f32,
    /// The edge must be seen this long (seconds) before it counts.
    pub hold_secs: f64,
    /// Frames moving faster than this (full-resolution px/frame) are motion-blurred: not judged.
    pub max_motion_px: f64,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            block: 8,
            flat_grad: 0.35,
            chroma_tol: 0.07,
            lum_ratio: 2.0,
            outer: 0.06,
            min_outer_plain: 0.5,
            line_frac: 0.5,
            min_depth: 0.08,
            match_lum_ratio: 1.3,
            max_novelty: 0.09,
            max_novelty_known: 0.25,
            learn_novelty: 0.04,
            learn_line_frac: 0.85,
            min_lum: 12.0,
            hold_secs: 0.3,
            max_motion_px: 12.0,
        }
    }
}

/// Appearance of a block or a band: chromaticity `(r-g, r+g-2b) / sum` and brightness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub chroma: [f32; 2],
    pub lum: f32,
}

impl Look {
    pub fn similar(&self, o: &Look, p: &EdgeParams, lum_ratio: f32) -> bool {
        let (dc0, dc1) = (self.chroma[0] - o.chroma[0], self.chroma[1] - o.chroma[1]);
        let r = (self.lum.max(1.0) / o.lum.max(1.0)).max(o.lum.max(1.0) / self.lum.max(1.0));
        dc0 * dc0 + dc1 * dc1 < p.chroma_tol * p.chroma_tol && r < lum_ratio
    }
}

#[derive(Debug, Clone, Copy)]
struct Block {
    look: Look,
    /// Mean gradient divided by brightness.
    grad: f32,
}

/// Per-block colour and contrast of a frame.
#[derive(Debug, Clone)]
pub struct BlockMap {
    bw: usize,
    bh: usize,
    blocks: Vec<Block>,
}

/// Side of the field the camera is heading to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Right,
    Left,
    Down,
    Up,
}

impl Side {
    pub fn from_heading(h: (f64, f64)) -> Self {
        if h.0 > 0.5 {
            Side::Right
        } else if h.0 < -0.5 {
            Side::Left
        } else if h.1 > 0.5 {
            Side::Down
        } else {
            Side::Up
        }
    }
}

impl BlockMap {
    /// `gray` is `frame` reduced `k` times (the analysis image); blocks are `block` px of `gray`.
    pub fn new(frame: &RgbImage, gray: &GrayImage, k: u32, block: u32) -> Self {
        let (gw, gh) = (gray.width() as usize, gray.height() as usize);
        let b = block.max(2) as usize;
        let (bw, bh) = (gw / b, gh / b);
        let g = gray.as_raw();
        let fb = b * k.max(1) as usize; // block side in frame pixels
        let step = (k as usize / 2).max(1);
        let (fw, fh) = (frame.width() as usize, frame.height() as usize);
        let px = frame.as_raw();
        let mut blocks = Vec::with_capacity(bw * bh);
        for by in 0..bh {
            for bx in 0..bw {
                let mut acc = [0u64; 3];
                let mut n = 0u64;
                for y in (by * fb..((by + 1) * fb).min(fh)).step_by(step) {
                    let row = y * fw * 3;
                    for x in (bx * fb..((bx + 1) * fb).min(fw)).step_by(step) {
                        let i = row + x * 3;
                        acc[0] += px[i] as u64;
                        acc[1] += px[i + 1] as u64;
                        acc[2] += px[i + 2] as u64;
                        n += 1;
                    }
                }
                let [r, gg, bb] = acc.map(|v| v as f32 / n.max(1) as f32);
                let s = r + gg + bb + 3.0;
                let lum = (r + gg + bb) / 3.0;
                let mut grad = 0u32;
                for y in by * b..(by + 1) * b {
                    for x in bx * b..(bx + 1) * b {
                        let v = g[y * gw + x] as i32;
                        if x + 1 < gw {
                            grad += (g[y * gw + x + 1] as i32 - v).unsigned_abs();
                        }
                        if y + 1 < gh {
                            grad += (g[(y + 1) * gw + x] as i32 - v).unsigned_abs();
                        }
                    }
                }
                blocks.push(Block {
                    look: Look { chroma: [(r - gg) / s, (r + gg - 2.0 * bb) / s], lum },
                    grad: grad as f32 / (b * b) as f32 / (lum + 4.0),
                });
            }
        }
        Self { bw, bh, blocks }
    }

    /// Number of lines (perpendicular to `side`) and their length.
    fn lines(&self, side: Side) -> (usize, usize) {
        match side {
            Side::Right | Side::Left => (self.bw, self.bh),
            Side::Down | Side::Up => (self.bh, self.bw),
        }
    }

    /// Block `j` of line `i`, lines counted from the `side` border inward.
    fn at(&self, side: Side, i: usize, j: usize) -> &Block {
        let (x, y) = match side {
            Side::Right => (self.bw - 1 - i, j),
            Side::Left => (i, j),
            Side::Down => (j, self.bh - 1 - i),
            Side::Up => (j, i),
        };
        &self.blocks[y * self.bw + x]
    }
}

/// Plain-block appearances sampled in the middle of the captured tiles (what the board looks like).
#[derive(Debug, Clone, Default)]
pub struct BoardModel {
    samples: Vec<Look>,
}

const MODEL_CAP: usize = 6000;

impl BoardModel {
    /// Adds the plain blocks of the central half of a captured tile, except those that look like a
    /// table already seen (an end-of-row tile may show the table up to its middle).
    pub fn add_tile(&mut self, map: &BlockMap, tables: &[Look], p: &EdgeParams) {
        self.add_tile_except(map, tables, p, |_, _| false);
    }

    /// Same, also leaving out the blocks for which `skip(x, y)` holds.
    fn add_tile_except(&mut self, map: &BlockMap, tables: &[Look], p: &EdgeParams, skip: impl Fn(usize, usize) -> bool) {
        for by in map.bh / 4..map.bh - map.bh / 4 {
            for bx in map.bw / 4..map.bw - map.bw / 4 {
                let b = &map.blocks[by * map.bw + bx];
                if !skip(bx, by)
                    && b.grad < p.flat_grad
                    && b.look.lum >= p.min_lum
                    && !tables.iter().any(|t| t.similar(&b.look, p, p.match_lum_ratio))
                {
                    self.samples.push(b.look);
                }
            }
        }
        if self.samples.len() > MODEL_CAP {
            // Keep a spread-out half: early and late tiles stay represented.
            let mut i = 0;
            self.samples.retain(|_| {
                i += 1;
                i % 2 == 0
            });
        }
    }

    /// Share of the board's plain blocks that look like `look`.
    pub fn match_frac(&self, look: &Look, p: &EdgeParams) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let n = self.samples.iter().filter(|s| s.similar(look, p, p.match_lum_ratio)).count();
        n as f32 / self.samples.len() as f32
    }

    /// Drops the samples that look like `table` (it leaked into the middle of a tile).
    pub fn purge(&mut self, table: &Look, p: &EdgeParams) {
        self.samples.retain(|s| !s.similar(table, p, p.match_lum_ratio));
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// A uniform band along the leading border.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    /// Depth from the border, as a fraction of the field.
    pub depth: f32,
    /// Share of the board model that looks like the band (low = it is not the board).
    pub novelty: f32,
    pub look: Look,
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Measures the uniform band along the `side` border; `None` when that border is busy.
pub fn measure(map: &BlockMap, side: Side, model: &BoardModel, p: &EdgeParams) -> Option<Band> {
    measure_lines(map, side, model, p, p.line_frac)
}

fn measure_lines(map: &BlockMap, side: Side, model: &BoardModel, p: &EdgeParams, line_frac: f32) -> Option<Band> {
    let (n, len) = map.lines(side);
    if n < 4 || len == 0 {
        return None;
    }
    let outer = ((n as f32 * p.outer).round() as usize).clamp(2, n);
    let plain: Vec<Look> = (0..outer)
        .flat_map(|i| (0..len).map(move |j| (i, j)))
        .map(|(i, j)| map.at(side, i, j))
        .filter(|b| b.grad < p.flat_grad)
        .map(|b| b.look)
        .collect();
    if (plain.len() as f32) < p.min_outer_plain * (outer * len) as f32 {
        return None;
    }
    let look = Look {
        chroma: [
            median(&mut plain.iter().map(|l| l.chroma[0]).collect::<Vec<_>>()),
            median(&mut plain.iter().map(|l| l.chroma[1]).collect::<Vec<_>>()),
        ],
        lum: median(&mut plain.iter().map(|l| l.lum).collect::<Vec<_>>()),
    };
    if look.lum < p.min_lum {
        return None;
    }
    let mut depth = 0;
    while depth < n {
        let like = (0..len)
            .filter(|&j| {
                let b = map.at(side, depth, j);
                b.grad < p.flat_grad && b.look.similar(&look, p, p.lum_ratio)
            })
            .count();
        if (like as f32) < line_frac * len as f32 {
            break;
        }
        depth += 1;
    }
    Some(Band { depth: depth as f32 / n as f32, novelty: model.match_frac(&look, p), look })
}

/// Light fall-off across the field (one side of the microscope image is much darker), as a
/// quadratic profile along x estimated from the brightness of the captured tiles. Only x: along y
/// the table often sits at the same place in every tile of a row, which would pass for darkness.
#[derive(Debug, Clone, Default)]
struct Light {
    cols: Vec<Vec<f32>>,
    gain_x: Vec<f32>,
}

/// Tiles needed before the profile is trusted (content averages out across tiles).
const LIGHT_MIN_TILES: usize = 3;

impl Light {
    fn add(&mut self, map: &BlockMap) {
        if self.cols.first().is_some_and(|c| c.len() != map.bw) {
            *self = Light::default();
        }
        let lum = |x: usize, y: usize| map.blocks[y * map.bw + x].look.lum;
        self.cols.push((0..map.bw).map(|x| median(&mut (0..map.bh).map(|y| lum(x, y)).collect::<Vec<_>>())).collect());
        if self.cols.len() >= LIGHT_MIN_TILES {
            self.gain_x = profile(&self.cols);
        }
    }

    /// Brightness corrected for the fall-off (same size as the tiles, else unchanged).
    fn correct(&self, map: &BlockMap) -> BlockMap {
        let mut out = map.clone();
        if self.gain_x.len() == map.bw {
            for y in 0..map.bh {
                for x in 0..map.bw {
                    out.blocks[y * map.bw + x].look.lum /= self.gain_x[x];
                }
            }
        }
        out
    }
}

/// Median over tiles of per-line medians, fitted by a quadratic and normalised to a mean of 1.
fn profile(lines: &[Vec<f32>]) -> Vec<f32> {
    let n = lines[0].len();
    let med: Vec<f32> = (0..n).map(|i| median(&mut lines.iter().map(|l| l[i]).collect::<Vec<_>>())).collect();
    // Least squares on t ∈ [-1, 1]: v = a + b t + c t².
    let ts: Vec<f64> = (0..n).map(|i| if n > 1 { 2.0 * i as f64 / (n - 1) as f64 - 1.0 } else { 0.0 }).collect();
    let mut m = [[0.0f64; 4]; 3];
    for (t, v) in ts.iter().zip(&med) {
        let f = [1.0, *t, t * t];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += f[r] * f[c];
            }
            m[r][3] += f[r] * *v as f64;
        }
    }
    // Gauss-Jordan elimination (3×3, well conditioned).
    for col in 0..3 {
        let piv = (col..3).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs())).unwrap_or(col);
        m.swap(col, piv);
        if m[col][col].abs() < 1e-9 {
            return vec![1.0; n];
        }
        for r in 0..3 {
            if r != col {
                let f = m[r][col] / m[col][col];
                for c in col..4 {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    let (a, b, c) = (m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]);
    let fit: Vec<f64> = ts.iter().map(|t| (a + b * t + c * t * t).max(1.0)).collect();
    let mean = fit.iter().sum::<f64>() / n as f64;
    fit.iter().map(|v| (v / mean).clamp(0.3, 3.0) as f32).collect()
}

/// What the detector reports for a frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct Status {
    /// Table band currently seen ahead (it passes the edge test; it may not be held long enough).
    pub band: Option<Band>,
    /// The edge has been seen long enough: act on it.
    pub edge: bool,
}

/// Learns what the board and the table look like, and turns per-frame measures into a stable
/// "board edge ahead" decision.
pub struct EdgeDetector {
    pub params: EdgeParams,
    model: BoardModel,
    light: Light,
    /// What the table looks like (learnt from tile borders and from acted-upon edges).
    tables: Vec<Look>,
    /// Appearances the user said are not the edge.
    rejected: Vec<Look>,
    since: Option<f64>,
    suppressed: bool,
    last: Option<Band>,
}

impl EdgeDetector {
    pub fn new(params: EdgeParams) -> Self {
        Self {
            params,
            model: BoardModel::default(),
            light: Light::default(),
            tables: Vec::new(),
            rejected: Vec::new(),
            since: None,
            suppressed: false,
            last: None,
        }
    }

    /// A tile was captured: learn what the board (its middle) and the table (a strongly novel band
    /// along one of its sides) look like, then look again.
    pub fn observe_tile(&mut self, map: &BlockMap) {
        self.light.add(map);
        let map = self.light.correct(map);
        let p = self.params.clone();
        // Candidate table bands: uniform along (nearly) the whole border. Their blocks stay out of
        // the board model (an end-of-row tile may show the table up to its middle) and they are
        // learnt as the table if the board model, without them, has nothing like them.
        let cands: Vec<(Side, usize, Look)> = [Side::Right, Side::Left, Side::Down, Side::Up]
            .into_iter()
            .filter_map(|side| {
                let b = measure_lines(&map, side, &BoardModel::default(), &p, p.learn_line_frac)?;
                let lines = (b.depth * map.lines(side).0 as f32).round() as usize;
                (b.depth >= p.min_depth && !self.is_rejected(&b.look)).then_some((side, lines, b.look))
            })
            .collect();
        let (bw, bh) = (map.bw, map.bh);
        let in_band = |x: usize, y: usize| {
            cands.iter().any(|&(side, d, _)| match side {
                Side::Right => x + d >= bw,
                Side::Left => x < d,
                Side::Down => y + d >= bh,
                Side::Up => y < d,
            })
        };
        self.model.add_tile_except(&map, &self.tables, &p, in_band);
        for (_, _, look) in &cands {
            if self.model.match_frac(look, &p) <= p.learn_novelty {
                self.learn_table(look);
            }
        }
        self.rearm();
    }

    fn learn_table(&mut self, look: &Look) {
        let p = &self.params;
        if !self.tables.iter().any(|t| t.similar(look, p, p.match_lum_ratio)) {
            self.tables.push(*look);
        }
        self.model.purge(look, p);
    }

    fn is_rejected(&self, look: &Look) -> bool {
        self.rejected.iter().any(|r| r.similar(look, &self.params, self.params.match_lum_ratio))
    }

    /// Does `look` match a table seen before?
    pub fn is_known_table(&self, look: &Look) -> bool {
        self.tables.iter().any(|t| t.similar(look, &self.params, self.params.match_lum_ratio))
    }

    /// The uniform band on `side` as measured (light-corrected), before any verdict.
    pub fn inspect(&self, map: &BlockMap, side: Side) -> Option<Band> {
        measure(&self.light.correct(map), side, &self.model, &self.params)
    }

    /// The table band on `side` of `map`, if there is one (judged against what was learnt; no timing).
    pub fn band(&self, map: &BlockMap, side: Side) -> Option<Band> {
        let p = &self.params;
        let b = self.inspect(map, side)?;
        let limit = if self.is_known_table(&b.look) { p.max_novelty_known } else { p.max_novelty };
        (b.depth >= p.min_depth && b.novelty <= limit && !self.is_rejected(&b.look)).then_some(b)
    }

    /// Judges a new frame. `map` is `None` when there is nothing to watch right now.
    pub fn update(&mut self, map: Option<&BlockMap>, side: Side, t: f64, motion_px: f64) -> Status {
        let Some(map) = map else {
            self.since = None;
            self.last = None;
            return Status::default();
        };
        if motion_px > self.params.max_motion_px {
            // Motion blur: keep the previous verdict, but do not let it mature.
            return Status { band: self.last, edge: false };
        }
        let band = self.band(map, side);
        self.last = band;
        if band.is_none() || self.suppressed {
            self.since = None;
            return Status { band, edge: false };
        }
        let since = *self.since.get_or_insert(t);
        Status { band, edge: t - since >= self.params.hold_secs }
    }

    /// The edge was acted upon: remember the table and start a fresh hold for the next one.
    pub fn accept(&mut self, band: &Band) {
        self.learn_table(&band.look);
        self.since = None;
    }

    /// The user said this is not the edge: never fire on that appearance again, and stay quiet
    /// until the next capture.
    pub fn reject(&mut self, look: &Look) {
        let p = self.params.clone();
        self.tables.retain(|t| !t.similar(look, &p, p.match_lum_ratio));
        self.rejected.push(*look);
        self.suppressed = true;
        self.since = None;
    }

    /// Re-arms after a capture.
    pub fn rearm(&mut self) {
        self.suppressed = false;
        self.since = None;
    }

    pub fn model(&self) -> &BoardModel {
        &self.model
    }

    /// Number of table appearances learnt so far.
    pub fn tables_known(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{synthetic_board, wood_table};
    use crate::tracking::gray_small;
    use image::Rgb;

    /// 960x540 view: board on the left of `edge_x`, wood table on the right, with a light
    /// fall-off toward the right like the real microscope.
    fn scene(edge_x: u32) -> RgbImage {
        let board = synthetic_board(960, 540);
        let table = wood_table(960, 540);
        RgbImage::from_fn(960, 540, |x, y| {
            let p = if x < edge_x { board.get_pixel(x, y) } else { table.get_pixel(x, y) };
            let f = 1.0 - 0.35 * x as f32 / 960.0;
            Rgb(p.0.map(|v| (v as f32 * f) as u8))
        })
    }

    fn map(img: &RgbImage) -> BlockMap {
        BlockMap::new(img, &gray_small(img, 2), 2, 8)
    }

    fn trained() -> EdgeDetector {
        let mut d = EdgeDetector::new(EdgeParams::default());
        d.observe_tile(&map(&scene(960)));
        d
    }

    #[test]
    fn textured_table_is_found_only_on_the_heading_side() {
        let d = trained();
        let m = map(&scene(700)); // table on the right 27 %
        let b = d.band(&m, Side::Right).expect("table not found");
        assert!((b.depth - 0.27).abs() < 0.05, "{b:?}");
        assert!(d.band(&m, Side::Left).is_none());
        assert!(d.band(&m, Side::Down).is_none());
        assert!(d.band(&map(&scene(960)), Side::Right).is_none());
    }

    #[test]
    fn plain_board_area_is_not_an_edge() {
        // A big solder-mask area on the right, the same colour as plain areas inside the board.
        let mask = Rgb([30u8, 100, 60]);
        let mut model_tile = scene(960);
        for y in 150..390 {
            for x in 300..660 {
                model_tile.put_pixel(x, y, mask);
            }
        }
        let mut d = EdgeDetector::new(EdgeParams::default());
        d.observe_tile(&map(&model_tile));
        let mut view = scene(960);
        for y in 0..540 {
            for x in 720..960 {
                view.put_pixel(x, y, mask);
            }
        }
        let raw = measure(&map(&view), Side::Right, d.model(), &d.params).expect("uniform band");
        assert!(raw.depth > 0.2, "{raw:?}");
        assert!(raw.novelty > 0.5, "{raw:?}");
        assert!(d.band(&map(&view), Side::Right).is_none());
    }

    #[test]
    fn edge_needs_hold_time_and_slow_motion() {
        let mut d = trained();
        let m = map(&scene(700));
        assert!(!d.update(Some(&m), Side::Right, 0.0, 2.0).edge);
        assert!(!d.update(Some(&m), Side::Right, 0.4, 40.0).edge); // fast: not judged
        assert!(d.update(Some(&m), Side::Right, 0.5, 2.0).edge);
        let busy = map(&scene(960));
        assert!(!d.update(Some(&busy), Side::Right, 0.6, 2.0).edge);
        assert!(!d.update(Some(&m), Side::Right, 0.7, 2.0).edge); // hold restarts
    }

    #[test]
    fn rejected_look_never_fires_again() {
        let mut d = trained();
        let m = map(&scene(700));
        let band = d.update(Some(&m), Side::Right, 0.0, 0.0).band.unwrap();
        d.reject(&band.look);
        d.rearm();
        for i in 0..20 {
            assert!(!d.update(Some(&m), Side::Right, i as f64 * 0.1, 0.0).edge);
        }
    }

    #[test]
    fn accepted_table_is_kept_out_of_the_board_model() {
        let mut d = trained();
        let m = map(&scene(400)); // end-of-row tile: table up to the middle
        let band = d.band(&m, Side::Right).unwrap();
        d.accept(&band);
        d.observe_tile(&m);
        assert!(d.band(&map(&scene(700)), Side::Right).is_some());
    }

    /// Real captures (downscaled 4×): wood table along the board edge, uneven light, soft right
    /// third. Every board edge must be found, and nothing else.
    #[test]
    fn real_tiles() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/edge");
        let load = |n: &str| image::open(dir.join(n)).expect("fixture").to_rgb8();
        // (file, sides that show the table); b2's top shows the board's plastic case, either way
        // is fine there.
        let a: &[(&str, &[Side])] = &[
            ("a1.jpg", &[Side::Left, Side::Up]),
            ("a2.jpg", &[Side::Up]),
            ("a3.jpg", &[Side::Up]),
            ("a5.jpg", &[Side::Up]),
            ("a7.jpg", &[Side::Up]),
        ];
        let b: &[(&str, &[Side])] =
            &[("b2.jpg", &[Side::Left]), ("b4.jpg", &[]), ("b6.jpg", &[Side::Right]), ("b8.jpg", &[]), ("b9.jpg", &[])];
        for set in [a, b] {
            let maps: Vec<(BlockMap, &[Side])> = set.iter().map(|(f, s)| (map4(&load(f)), *s)).collect();
            let mut d = EdgeDetector::new(EdgeParams::default());
            for (m, _) in &maps {
                d.observe_tile(m);
            }
            for ((m, sides), (name, _)) in maps.iter().zip(set) {
                for side in [Side::Right, Side::Left, Side::Down, Side::Up] {
                    let found = d.band(m, side);
                    let raw = d.inspect(m, side);
                    eprintln!("{name} {side:?}: {:?} known={}", raw.map(|b| (b.depth, b.novelty, b.look)), raw.is_some_and(|b| d.is_known_table(&b.look)));
                    if *name == "b2.jpg" && side == Side::Up {
                        continue;
                    }
                    assert_eq!(found.is_some(), sides.contains(&side), "{name} {side:?}: {found:?}");
                }
            }
        }
    }

    /// The fixtures are already the analysis image (k = 4 of 1920x1080).
    fn map4(img: &RgbImage) -> BlockMap {
        BlockMap::new(img, &gray_small(img, 1), 1, 8)
    }
}

