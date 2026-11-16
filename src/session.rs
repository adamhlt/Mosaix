//! Session saving: full-resolution tiles (PNG) + JSON metadata.

use anyhow::{Context, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileMeta {
    pub file: String,
    /// Top-left corner position (full-resolution px, board coordinates).
    pub x: f64,
    pub y: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SessionMeta {
    pub version: u32,
    pub tiles: Vec<TileMeta>,
}

pub struct Session {
    dir: PathBuf,
    pub meta: SessionMeta,
    next_id: usize,
}

impl Session {
    /// Creates `root/<local date and time>/` (`2026-11-16_19-34-59`, `-2`, `-3`… when one already
    /// exists) with its `tiles/` subfolder. The names sort in the order the sessions were made.
    pub fn create(root: &Path) -> Result<Self> {
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let dir = std::iter::once(root.join(&stamp))
            .chain((2..).map(|n| root.join(format!("{stamp}-{n}"))))
            .find(|d| !d.exists())
            .expect("an unused name exists");
        std::fs::create_dir_all(dir.join("tiles")).with_context(|| format!("creating {}", dir.display()))?;
        let s = Self { dir, meta: SessionMeta { version: 1, tiles: vec![] }, next_id: 1 };
        s.write_meta()?;
        Ok(s)
    }

    /// Reopens an existing session (resume).
    pub fn open(dir: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(dir.join("session.json"))?;
        let meta: SessionMeta = serde_json::from_str(&txt)?;
        let next_id = meta.tiles.len() + 1;
        Ok(Self { dir: dir.to_path_buf(), meta, next_id })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn write_meta(&self) -> Result<()> {
        // Atomic write: temporary file, then rename
        let tmp = self.dir.join("session.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.meta)?)?;
        std::fs::rename(&tmp, self.dir.join("session.json"))?;
        Ok(())
    }

    pub fn add_tile(&mut self, img: &RgbImage, pos: (f64, f64)) -> Result<()> {
        let file = format!("tiles/{:05}.png", self.next_id);
        self.next_id += 1;
        img.save(self.dir.join(&file))?;
        self.meta.tiles.push(TileMeta { file, x: pos.0, y: pos.1, width: img.width(), height: img.height() });
        self.write_meta()
    }

    /// Undoes the last tile (removes the file and its metadata).
    pub fn remove_last(&mut self) -> Result<Option<TileMeta>> {
        let Some(t) = self.meta.tiles.pop() else { return Ok(None) };
        let _ = std::fs::remove_file(self.dir.join(&t.file));
        self.write_meta()?;
        Ok(Some(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn save_reload_undo() {
        let root = std::env::temp_dir().join(format!("mosaix-test-{}", std::process::id()));
        let mut s = Session::create(&root).unwrap();
        let img = RgbImage::from_pixel(32, 16, Rgb([1, 2, 3]));
        s.add_tile(&img, (10.5, -4.0)).unwrap();
        s.add_tile(&img, (20.0, 0.0)).unwrap();
        let r = Session::open(s.dir()).unwrap();
        assert_eq!(r.meta.tiles.len(), 2);
        assert_eq!(r.meta.tiles[0].x, 10.5);
        assert!(s.dir().join("tiles/00001.png").exists());
        s.remove_last().unwrap();
        assert!(!s.dir().join("tiles/00002.png").exists());
        assert_eq!(Session::open(s.dir()).unwrap().meta.tiles.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sessions_are_named_by_date_and_never_collide() {
        let root = std::env::temp_dir().join(format!("mosaix-names-{}", std::process::id()));
        let a = Session::create(&root).unwrap();
        let b = Session::create(&root).unwrap();
        assert_ne!(a.dir(), b.dir());
        let name = a.dir().file_name().unwrap().to_string_lossy().to_string();
        // 2026-11-16_19-34-59
        assert_eq!(name.len(), 19, "{name}");
        assert!(name.chars().enumerate().all(|(i, c)| if [4, 7, 13, 16].contains(&i) {
            c == '-'
        } else if i == 10 {
            c == '_'
        } else {
            c.is_ascii_digit()
        }));
        let _ = std::fs::remove_dir_all(&root);
    }
}
