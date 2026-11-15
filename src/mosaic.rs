//! Tiles, positions and preview canvas of the mosaic.

use crate::tracking::rgb_downscale;
use image::{imageops, RgbImage, RgbaImage};

/// Downscale factor of the thumbnails kept in memory (1/8 of full resolution).
pub const THUMB_K: u32 = 8;

pub struct MosaicTile {
    /// Top-left corner position (full-resolution px, board coordinates).
    pub pos: (f64, f64),
    /// Full-resolution size.
    pub size: (u32, u32),
    pub thumb: RgbImage,
}

/// Rendered preview: `canvas_px = (board_px - origin) * scale`.
pub struct Preview {
    pub image: RgbaImage,
    pub scale: f64,
    pub origin: (f64, f64),
    /// Revision number (incremented on every render).
    pub rev: u64,
}

#[derive(Default)]
pub struct Mosaic {
    tiles: Vec<MosaicTile>,
    rev: u64,
}

impl Mosaic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
    pub fn tiles(&self) -> &[MosaicTile] {
        &self.tiles
    }

    pub fn add(&mut self, full: &RgbImage, pos: (f64, f64)) {
        self.tiles.push(MosaicTile { pos, size: (full.width(), full.height()), thumb: rgb_downscale(full, THUMB_K) });
    }

    pub fn add_thumb(&mut self, thumb: RgbImage, pos: (f64, f64), size: (u32, u32)) {
        self.tiles.push(MosaicTile { pos, size, thumb });
    }

    pub fn pop(&mut self) -> Option<MosaicTile> {
        self.tiles.pop()
    }

    /// Bounding box of all tiles: `(x0, y0, x1, y1)` in full-resolution px.
    pub fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        let mut it = self.tiles.iter();
        let t = it.next()?;
        let mut b = (t.pos.0, t.pos.1, t.pos.0 + t.size.0 as f64, t.pos.1 + t.size.1 as f64);
        for t in it {
            b.0 = b.0.min(t.pos.0);
            b.1 = b.1.min(t.pos.1);
            b.2 = b.2.max(t.pos.0 + t.size.0 as f64);
            b.3 = b.3.max(t.pos.1 + t.size.1 as f64);
        }
        Some(b)
    }

    /// Renders the canvas (simple placement, no blending: the last tile covers the others),
    /// downscaled to fit within `max_side` pixels.
    pub fn render(&mut self, max_side: u32) -> Option<Preview> {
        let (x0, y0, x1, y1) = self.bounds()?;
        let thumb_scale = 1.0 / THUMB_K as f64;
        let (cw, ch) = (((x1 - x0) * thumb_scale).ceil() as u32 + 1, ((y1 - y0) * thumb_scale).ceil() as u32 + 1);
        let mut canvas = RgbaImage::from_pixel(cw, ch, image::Rgba([24, 24, 28, 255]));
        for t in &self.tiles {
            let px = ((t.pos.0 - x0) * thumb_scale).round() as i64;
            let py = ((t.pos.1 - y0) * thumb_scale).round() as i64;
            let rgba = RgbaImage::from_fn(t.thumb.width(), t.thumb.height(), |x, y| {
                let p = t.thumb.get_pixel(x, y).0;
                image::Rgba([p[0], p[1], p[2], 255])
            });
            imageops::replace(&mut canvas, &rgba, px, py);
        }
        let mut scale = thumb_scale;
        let side = cw.max(ch);
        if side > max_side {
            let f = max_side as f64 / side as f64;
            let (nw, nh) = (((cw as f64 * f) as u32).max(1), ((ch as f64 * f) as u32).max(1));
            canvas = imageops::resize(&canvas, nw, nh, imageops::FilterType::Triangle);
            scale *= f;
        }
        self.rev += 1;
        Some(Preview { image: canvas, scale, origin: (x0, y0), rev: self.rev })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn bounds_and_render() {
        let mut m = Mosaic::new();
        assert!(m.render(512).is_none());
        let a = RgbImage::from_pixel(160, 80, Rgb([255, 0, 0]));
        m.add(&a, (0.0, 0.0));
        m.add(&a, (100.0, 40.0));
        assert_eq!(m.bounds(), Some((0.0, 0.0, 260.0, 120.0)));
        let p = m.render(512).unwrap();
        assert!((p.scale - 1.0 / 8.0).abs() < 1e-9);
        assert!(p.image.width() >= 32 && p.image.height() >= 15);
        // A pixel of the first tile is red
        assert_eq!(p.image.get_pixel(2, 2).0, [255, 0, 0, 255]);
        m.pop();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn render_downscales_to_fit() {
        let mut m = Mosaic::new();
        let a = RgbImage::from_pixel(1600, 800, Rgb([0, 255, 0]));
        m.add(&a, (0.0, 0.0));
        let p = m.render(100).unwrap();
        assert!(p.image.width() <= 100 && p.image.height() <= 100);
        assert!(p.scale < 1.0 / 8.0);
    }
}
