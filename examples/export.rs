//! Headless export without the UI (fast to build):
//! `cargo run --release --no-default-features --example export -- <session_dir> <out> [--scale 0.5]`

use mosaix::export::{export_session, ExportOptions, Progress};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(dir), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("usage: export <session_dir> <out.png|jpg|tif> [--scale 0.5] [--crop] [--no-refine] [--no-brightness]");
        std::process::exit(2);
    };
    let mut opts = ExportOptions::default();
    if let Some(i) = args.iter().position(|a| a == "--scale") {
        opts.scale = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    }
    opts.crop = args.iter().any(|a| a == "--crop");
    opts.refine = !args.iter().any(|a| a == "--no-refine");
    opts.match_brightness = !args.iter().any(|a| a == "--no-brightness");
    match export_session(std::path::Path::new(dir), std::path::Path::new(out), &opts, &Progress::new(None)) {
        Ok(r) => println!(
            "{}x{} px, {} tiles, {} pairs, max correction {:.1} px, overlap error {}",
            r.width,
            r.height,
            r.tiles,
            r.pairs_used,
            r.max_correction,
            r.overlap_error.map_or("n/a".into(), |e| format!("{e:.2}"))
        ),
        Err(e) => {
            eprintln!("export failed: {e:#}");
            std::process::exit(1);
        }
    }
}
