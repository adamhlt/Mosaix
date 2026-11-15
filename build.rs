//! Windows: embeds assets/icon/icon.ico as the executable's icon (resource id 1), which
//! Explorer shows and the app's windows load (`rune::set_app_icon_resource(1)`).

fn main() {
    println!("cargo:rerun-if-changed=assets/icon/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon_with_id("assets/icon/icon.ico", "1");
        res.set("FileDescription", "Mosaïx");
        res.set("ProductName", "Mosaïx");
        if let Err(e) = res.compile() {
            panic!("couldn't embed the Windows icon: {e}");
        }
    }
}
