//! Sharpness measure: variance of the Laplacian.

use image::GrayImage;

/// Variance of the 3x3 Laplacian (4-neighbour kernel). The larger, the sharper the image.
/// Values are only comparable between images of the same size/scale.
pub fn sharpness(img: &GrayImage) -> f32 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let p = img.as_raw();
    let (mut sum, mut sum2) = (0.0f64, 0.0f64);
    for y in 1..h - 1 {
        let row = y * w;
        for x in 1..w - 1 {
            let l = p[row + x - 1] as i32 + p[row + x + 1] as i32 + p[row - w + x] as i32 + p[row + w + x] as i32 - 4 * p[row + x] as i32;
            let l = l as f64;
            sum += l;
            sum2 += l * l;
        }
    }
    let n = ((w - 2) * (h - 2)) as f64;
    let mean = sum / n;
    (sum2 / n - mean * mean).max(0.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops, GrayImage, Luma};

    fn checker(n: u32, cell: u32) -> GrayImage {
        GrayImage::from_fn(n, n, |x, y| if (x / cell + y / cell).is_multiple_of(2) { Luma([30]) } else { Luma([220]) })
    }

    #[test]
    fn flat_image_has_zero_sharpness() {
        let img = GrayImage::from_pixel(64, 64, Luma([128]));
        assert_eq!(sharpness(&img), 0.0);
    }

    #[test]
    fn blur_reduces_sharpness_monotonically() {
        let sharp = checker(128, 4);
        let b1 = imageops::blur(&sharp, 1.0);
        let b2 = imageops::blur(&sharp, 2.5);
        let (s0, s1, s2) = (sharpness(&sharp), sharpness(&b1), sharpness(&b2));
        assert!(s0 > s1 && s1 > s2, "{s0} {s1} {s2}");
    }

    #[test]
    fn tiny_image_is_safe() {
        assert_eq!(sharpness(&GrayImage::new(2, 2)), 0.0);
    }
}
