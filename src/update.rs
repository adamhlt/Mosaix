//! Updates from GitHub releases.
//!
//! At startup (and when the version in the status bar is clicked) the latest release is read
//! from the GitHub API. When it is newer, the app offers to update itself where it can:
//!
//! | Install                              | Asset                       | How it is replaced              |
//! | ------------------------------------ | --------------------------- | ------------------------------- |
//! | Windows, installed with the setup    | `*-setup.exe`               | the installer runs silently     |
//! | Windows, portable                    | `*-portable.zip`            | `mosaix.exe` swapped in place   |
//! | macOS, run from `Mosaix.app`         | `*.dmg`                     | the bundle is swapped in place  |
//! | Linux x86-64, a binary somewhere     | `*.tar.gz`                  | the binary is swapped in place  |
//!
//! Anywhere else (another CPU, a build from source, a folder that can't be written, a failed
//! download) the offer sends the user to the release page instead.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use rune::{Tx, Waker};
use sha2::{Digest, Sha256};

const REPO: &str = "adamhlt/Mosaix";
/// The release's version: its tag when the release workflow built it (`MOSAIX_VERSION`), else
/// Cargo.toml's.
pub const VERSION: &str = match option_env!("MOSAIX_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

// ---- What a release is ------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
    /// Lowercase hex SHA-256, when GitHub reports it.
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub tag: String,
    pub version: String,
    pub assets: Vec<Asset>,
}

impl Release {
    /// The release's own page. Built from the tag rather than read from the response, so it can
    /// only ever point at this repository.
    pub fn page(&self) -> String {
        let tag: String = self.tag.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')).collect();
        format!("https://github.com/{REPO}/releases/tag/{tag}")
    }
}

/// How this copy of the app can be replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Windows, installed with the setup: run the new setup.
    Installer,
    /// Windows, unzipped somewhere: swap `mosaix.exe`.
    Portable,
    /// macOS, in an app bundle: swap the bundle.
    MacApp,
    /// Linux: swap the executable.
    Binary,
    /// Nothing to download for this install: open the release page.
    Manual,
}

impl Method {
    fn suffix(self) -> &'static str {
        match self {
            Method::Installer => "-setup.exe",
            Method::Portable => "-portable.zip",
            Method::MacApp => ".dmg",
            Method::Binary => ".tar.gz",
            Method::Manual => "",
        }
    }

    fn os_token(self) -> &'static str {
        match self {
            Method::Installer | Method::Portable => "windows",
            Method::MacApp => "macos",
            Method::Binary => "linux",
            Method::Manual => "",
        }
    }
}

/// A newer release, and what can be done about it here.
#[derive(Clone, Debug, PartialEq)]
pub struct Offer {
    pub release: Release,
    pub method: Method,
    /// The file to download; none when the method is manual or the release lacks it.
    pub asset: Option<Asset>,
}

impl Offer {
    pub fn can_install(&self) -> bool {
        self.method != Method::Manual && self.asset.is_some()
    }
}

// ---- Versions ---------------------------------------------------------------------------------

/// `v1.2.3`, `1.2` or `1.2.3-beta.1` as `[major, minor, patch]`; a pre-release suffix is dropped.
fn parse_version(v: &str) -> Option<[u64; 3]> {
    let core = v.trim().trim_start_matches(['v', 'V']).split(['-', '+']).next()?;
    let mut out = [0u64; 3];
    let mut parts = core.split('.');
    for slot in &mut out {
        match parts.next() {
            Some(p) => *slot = p.parse().ok()?,
            None => break,
        }
    }
    Some(out)
}

pub fn is_newer(remote: &str, local: &str) -> bool {
    match (parse_version(remote), parse_version(local)) {
        (Some(r), Some(l)) => r > l,
        _ => false,
    }
}

fn parse_release(v: &serde_json::Value) -> Option<Release> {
    let tag = v.get("tag_name")?.as_str()?.to_string();
    let assets = v
        .get("assets")?
        .as_array()?
        .iter()
        .filter_map(|a| {
            Some(Asset {
                name: a.get("name")?.as_str()?.to_string(),
                url: a.get("browser_download_url")?.as_str()?.to_string(),
                size: a.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                sha256: a.get("digest").and_then(|d| d.as_str()).and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase),
            })
        })
        .collect();
    Some(Release { version: tag.trim_start_matches(['v', 'V']).to_string(), tag, assets })
}

// ---- Where this copy runs ---------------------------------------------------------------------

/// The `.app` folder holding `exe` (`Mosaix.app/Contents/MacOS/mosaix`), if it is in one.
fn mac_bundle(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let bundle = contents.parent()?;
    let named = macos.file_name()? == "MacOS" && contents.file_name()? == "Contents";
    (named && bundle.extension().is_some_and(|e| e == "app")).then(|| bundle.to_path_buf())
}

/// Whether `exe` was built by Cargo from a source tree (`<project>/target/<profile>/mosaix`).
fn is_cargo_build(exe: &Path) -> bool {
    exe.ancestors()
        .skip(1)
        .any(|d| d.file_name().is_some_and(|n| n == "target") && d.parent().is_some_and(|p| p.join("Cargo.toml").is_file()))
}

/// Whether the Inno Setup uninstaller sits beside `exe`: the app came from the setup.
fn installed_by_setup(exe: &Path) -> bool {
    let Some(dir) = exe.parent() else { return false };
    fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_lowercase();
                n.starts_with("unins") && n.ends_with(".exe")
            })
        })
        .unwrap_or(false)
}

/// How a running copy at `exe` can be updated, on `os` and `arch` (`std::env::consts` values).
pub fn detect_method(exe: &Path, os: &str, arch: &str) -> Method {
    if is_cargo_build(exe) {
        return Method::Manual;
    }
    match os {
        // Only x86-64 builds are published for Windows and Linux.
        "windows" if arch == "x86_64" => {
            if installed_by_setup(exe) {
                Method::Installer
            } else {
                Method::Portable
            }
        }
        "linux" if arch == "x86_64" => Method::Binary,
        "macos" if mac_bundle(exe).is_some() => Method::MacApp,
        _ => Method::Manual,
    }
}

fn pick_asset(method: Method, assets: &[Asset]) -> Option<Asset> {
    if method == Method::Manual {
        return None;
    }
    assets
        .iter()
        .find(|a| {
            let n = a.name.to_lowercase();
            n.starts_with("mosaix") && n.ends_with(method.suffix()) && n.contains(method.os_token())
        })
        .cloned()
}

// ---- State ------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Idle,
    Checking,
    Available(Offer),
    Downloading {
        offer: Offer,
        done: u64,
        total: u64,
    },
    Installing(Offer),
    /// The update didn't happen; the release page is still one click away.
    Failed {
        offer: Offer,
        message: String,
    },
}

enum Msg {
    Checked { manual: bool, result: Result<Option<Offer>, String> },
    Installing,
    Restart,
    Failed(String),
}

pub struct Updater {
    pub state: State,
    rx: Option<Receiver<Msg>>,
    done: Arc<AtomicU64>,
    /// A line for the status bar, taken by the window.
    pub notice: Option<String>,
    /// Wakes the window when the background thread has news: an idle window draws nothing,
    /// so without it the progress would only show on the next mouse move.
    waker: Option<Waker>,
}

fn wake(w: &Option<Waker>) {
    if let Some(w) = w {
        w();
    }
}

impl Updater {
    pub fn new() -> Updater {
        Updater { state: State::Idle, rx: None, done: Arc::new(AtomicU64::new(0)), notice: None, waker: None }
    }

    pub fn set_waker(&mut self, w: Waker) {
        self.waker = Some(w);
    }

    fn busy(&self) -> bool {
        matches!(self.state, State::Checking | State::Downloading { .. } | State::Installing(_))
    }

    /// Looks for a newer release in the background. `manual` also reports "up to date" and
    /// errors, where the check at startup stays silent.
    pub fn check(&mut self, manual: bool) {
        if self.busy() {
            return;
        }
        if !manual && std::env::var_os("MOSAIX_NO_UPDATE_CHECK").is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.state = State::Checking;
        let waker = self.waker.clone();
        std::thread::spawn(move || {
            let result = fetch_latest().map(|release| {
                if !is_newer(&release.version, VERSION) {
                    return None;
                }
                let method =
                    std::env::current_exe().map_or(Method::Manual, |exe| detect_method(&exe, std::env::consts::OS, std::env::consts::ARCH));
                let asset = pick_asset(method, &release.assets);
                // A release without a file for this install is still worth announcing.
                let method = if asset.is_some() { method } else { Method::Manual };
                Some(Offer { release, method, asset })
            });
            let _ = tx.send(Msg::Checked { manual, result });
            wake(&waker);
        });
    }

    /// Downloads and applies the offered update.
    pub fn install(&mut self) {
        let (State::Available(offer) | State::Failed { offer, .. }) = self.state.clone() else { return };
        let Some(asset) = offer.asset.clone().filter(|_| offer.can_install()) else { return };
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.done.store(0, Ordering::Relaxed);
        self.state = State::Downloading { done: 0, total: asset.size, offer: offer.clone() };
        let done = self.done.clone();
        let waker = self.waker.clone();
        std::thread::spawn(move || {
            let msg = match run_install(&offer, &asset, &done, &tx, &waker) {
                Ok(()) => Msg::Restart,
                Err(e) => Msg::Failed(e),
            };
            let _ = tx.send(msg);
            wake(&waker);
        });
    }

    pub fn dismiss(&mut self) {
        if !self.busy() {
            self.state = State::Idle;
        }
    }

    /// The release page for the current offer.
    pub fn page(&self) -> Option<String> {
        match &self.state {
            State::Available(o) | State::Installing(o) | State::Downloading { offer: o, .. } | State::Failed { offer: o, .. } => {
                Some(o.release.page())
            }
            _ => None,
        }
    }

    /// Called from the window's poll: takes what the background thread reported. True when the
    /// view needs to run again.
    pub fn poll(&mut self, tx: &mut Tx) -> bool {
        let mut changed = false;
        while let Some(msg) = self.rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            changed = true;
            match msg {
                Msg::Checked { manual, result } => {
                    self.state = State::Idle;
                    match result {
                        Ok(Some(offer)) => self.state = State::Available(offer),
                        Ok(None) if manual => self.notice = Some(format!("Mosaïx {VERSION} is up to date.")),
                        Err(e) if manual => self.notice = Some(format!("Couldn't check for updates: {e}")),
                        _ => {}
                    }
                }
                Msg::Installing => {
                    if let State::Downloading { offer, .. } = &self.state {
                        self.state = State::Installing(offer.clone());
                    }
                }
                Msg::Restart => {
                    tx.close_window();
                }
                Msg::Failed(message) => {
                    if let State::Downloading { offer, .. } | State::Installing(offer) = self.state.clone() {
                        self.state = State::Failed { offer, message };
                    }
                }
            }
        }
        if let State::Downloading { done, total, .. } = &mut self.state {
            let now = self.done.load(Ordering::Relaxed);
            if now != *done {
                *done = now;
                changed = true;
            }
            *total = (*total).max(now);
        }
        if self.busy() {
            tx.wake_after(Duration::from_millis(150));
        }
        changed
    }
}

// ---- Network ----------------------------------------------------------------------------------

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(timeout)).user_agent(concat!("mosaix/", env!("CARGO_PKG_VERSION"))).build().into()
}

fn fetch_latest() -> Result<Release, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let mut resp =
        agent(Duration::from_secs(20)).get(&url).header("Accept", "application/vnd.github+json").call().map_err(|e| match e {
            // GitHub answers 404 while the repository has no release yet.
            ureq::Error::StatusCode(404) => "no release has been published yet".to_string(),
            e => e.to_string(),
        })?;
    let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    parse_release(&json).ok_or_else(|| "unexpected answer from GitHub".to_string())
}

fn download(asset: &Asset, dest: &Path, done: &AtomicU64, waker: &Option<Waker>) -> Result<(), String> {
    // Only ever fetch from GitHub, whatever the answer said.
    if !asset.url.starts_with(&format!("https://github.com/{REPO}/releases/download/")) {
        return Err("unexpected download address".into());
    }
    let resp = agent(Duration::from_secs(900)).get(&asset.url).call().map_err(|e| e.to_string())?;
    let mut file = File::create(dest).map_err(|e| e.to_string())?;
    let (total, hash) = copy_counting(&mut resp.into_body().into_reader(), &mut file, asset.size, done, &|| wake(waker))?;
    file.flush().map_err(|e| e.to_string())?;
    if asset.size > 0 && total != asset.size {
        return Err(format!("the download is incomplete ({total} of {} bytes)", asset.size));
    }
    if let Some(want) = &asset.sha256 {
        let got: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        if &got != want {
            return Err("the download doesn't match its checksum".into());
        }
    }
    Ok(())
}

/// Copies `from` into `to`, counting the bytes in `done` and hashing them, and calls `wake`
/// each whole percent of `size` (or every 256 KB when the size is unknown).
fn copy_counting(
    from: &mut impl Read,
    to: &mut impl Write,
    size: u64,
    done: &AtomicU64,
    wake: &dyn Fn(),
) -> Result<(u64, Vec<u8>), String> {
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    let step = (size / 100).clamp(1, 256 * 1024);
    loop {
        let n = from.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        to.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hash.update(&buf[..n]);
        let before = total;
        total += n as u64;
        done.store(total, Ordering::Relaxed);
        if before / step != total / step {
            wake();
        }
    }
    Ok((total, hash.finalize().to_vec()))
}

// ---- Applying ---------------------------------------------------------------------------------

fn run_install(offer: &Offer, asset: &Asset, done: &AtomicU64, tx: &Sender<Msg>, waker: &Option<Waker>) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let work = std::env::temp_dir().join(format!("mosaix-update-{}", offer.release.version));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let archive = work.join(&asset.name);
    let result = download(asset, &archive, done, waker).and_then(|()| {
        let _ = tx.send(Msg::Installing);
        wake(waker);
        apply(offer.method, &archive, &exe, &work)
    });
    // The setup runs after this process exits, so it has to stay.
    if result.is_err() || offer.method != Method::Installer {
        let _ = fs::remove_dir_all(&work);
    }
    result
}

fn apply(method: Method, archive: &Path, exe: &Path, work: &Path) -> Result<(), String> {
    match method {
        Method::Installer => run_setup(archive, exe),
        Method::Portable => {
            replace_from_zip(archive, exe)?;
            relaunch(exe)
        }
        Method::Binary => {
            replace_from_tar_gz(archive, exe)?;
            relaunch(exe)
        }
        Method::MacApp => {
            let bundle = mac_bundle(exe).ok_or("not running from an app bundle")?;
            replace_mac_app(archive, &bundle, work)?;
            relaunch(&bundle)
        }
        Method::Manual => Err("this install can't update itself".into()),
    }
}

/// `mosaix.exe` → `mosaix.exe.new`.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Puts `new` where `target` is: the old one is moved aside first, which works on a running
/// executable (Windows) and is undone if the second step fails.
fn swap(new: &Path, target: &Path) -> Result<(), String> {
    let old = sibling(target, ".old");
    let _ = fs::remove_file(&old);
    let _ = fs::remove_dir_all(&old);
    fs::rename(target, &old).map_err(|e| format!("couldn't replace {}: {e}", target.display()))?;
    if let Err(e) = fs::rename(new, target) {
        let _ = fs::rename(&old, target);
        return Err(format!("couldn't replace {}: {e}", target.display()));
    }
    // A running executable can't be deleted on Windows: `cleanup` removes it next time.
    let _ = fs::remove_file(&old);
    let _ = fs::remove_dir_all(&old);
    Ok(())
}

fn replace_from_zip(archive: &Path, exe: &Path) -> Result<(), String> {
    let mut zip = zip::ZipArchive::new(File::open(archive).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let name = exe.file_name().and_then(|n| n.to_str()).unwrap_or("mosaix.exe").to_string();
    let new = sibling(exe, ".new");
    let mut entry = zip.by_name(&name).map_err(|_| format!("{name} is missing from the archive"))?;
    let mut out = File::create(&new).map_err(|e| format!("couldn't write next to {}: {e}", exe.display()))?;
    let copied = io::copy(&mut entry, &mut out).map_err(|e| e.to_string());
    drop(out);
    if let Err(e) = copied {
        let _ = fs::remove_file(&new);
        return Err(e);
    }
    swap(&new, exe).inspect_err(|_| {
        let _ = fs::remove_file(&new);
    })
}

fn replace_from_tar_gz(archive: &Path, exe: &Path) -> Result<(), String> {
    let gz = flate2::read::GzDecoder::new(File::open(archive).map_err(|e| e.to_string())?);
    let mut tar = tar::Archive::new(gz);
    let name = exe.file_name().map(|n| n.to_os_string()).unwrap_or_else(|| "mosaix".into());
    let new = sibling(exe, ".new");
    for entry in tar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let is_exe =
            entry.header().entry_type().is_file() && entry.path().is_ok_and(|p| p.file_name().is_some_and(|n| n == name.as_os_str()));
        if !is_exe {
            continue;
        }
        let mut out = File::create(&new).map_err(|e| format!("couldn't write next to {}: {e}", exe.display()))?;
        let copied = io::copy(&mut entry, &mut out).map_err(|e| e.to_string());
        drop(out);
        if let Err(e) = copied {
            let _ = fs::remove_file(&new);
            return Err(e);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&new, fs::Permissions::from_mode(0o755));
        }
        // Renaming over the running file is allowed on Unix, and atomic.
        return fs::rename(&new, exe).map_err(|e| {
            let _ = fs::remove_file(&new);
            format!("couldn't replace {}: {e}", exe.display())
        });
    }
    Err("the executable is missing from the archive".into())
}

fn replace_mac_app(dmg: &Path, bundle: &Path, work: &Path) -> Result<(), String> {
    let mount = work.join("mount");
    fs::create_dir_all(&mount).map_err(|e| e.to_string())?;
    run(
        "hdiutil",
        &[
            "attach".as_ref(),
            "-nobrowse".as_ref(),
            "-readonly".as_ref(),
            "-noverify".as_ref(),
            "-mountpoint".as_ref(),
            mount.as_os_str(),
            dmg.as_os_str(),
        ],
    )?;
    let staged = sibling(bundle, ".new");
    let copied = (|| {
        let _ = fs::remove_dir_all(&staged);
        let app = fs::read_dir(&mount)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "app"))
            .ok_or("the disk image holds no app")?;
        // `ditto` keeps the bundle's signature, symlinks and permissions intact.
        run("ditto", &[app.as_os_str(), staged.as_os_str()]).map_err(|e| format!("couldn't write next to {}: {e}", bundle.display()))
    })();
    let _ = run("hdiutil", &["detach".as_ref(), "-force".as_ref(), mount.as_os_str()]);
    copied?;
    swap(&staged, bundle).inspect_err(|_| {
        let _ = fs::remove_dir_all(&staged);
    })
}

fn run(program: &str, args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let out = Command::new(program).args(args).stdin(Stdio::null()).output().map_err(|e| format!("{program}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("{program}: {}", String::from_utf8_lossy(&out.stderr).trim()))
    }
}

// ---- Restarting -------------------------------------------------------------------------------

/// Quotes a path for a single-quoted PowerShell string.
fn ps_quote(p: &Path) -> String {
    format!("'{}'", p.to_string_lossy().replace('\'', "''"))
}

/// Runs `script` in a hidden PowerShell that outlives this process.
fn spawn_powershell(script: &str) -> Result<(), String> {
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-Command", script])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        cmd.creation_flags(0x0800_0000 | 0x0000_0200);
    }
    cmd.spawn().map(drop).map_err(|e| format!("powershell: {e}"))
}

/// Runs the new setup once this process has exited, then starts the app again.
fn run_setup(setup: &Path, exe: &Path) -> Result<(), String> {
    let script = format!(
        "Wait-Process -Id {} -ErrorAction SilentlyContinue; \
         Start-Process -Wait -FilePath {} -ArgumentList '/SILENT','/NORESTART','/SUPPRESSMSGBOXES'; \
         Start-Process -FilePath {}",
        std::process::id(),
        ps_quote(setup),
        ps_quote(exe),
    );
    spawn_powershell(&script)
}

/// Starts `target` (an executable, or on macOS an app bundle) once this process has exited.
fn relaunch(target: &Path) -> Result<(), String> {
    if cfg!(windows) {
        let script =
            format!("Wait-Process -Id {} -ErrorAction SilentlyContinue; Start-Process -FilePath {}", std::process::id(), ps_quote(target));
        return spawn_powershell(&script);
    }
    // $0 is the target and $1 this process's id.
    let script = r#"while kill -0 "$1" 2>/dev/null; do sleep 0.2; done; if [ -d "$0" ]; then open -n "$0"; else exec "$0"; fi"#;
    Command::new("sh")
        .args(["-c", script])
        .arg(target)
        .arg(std::process::id().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(drop)
        .map_err(|e| format!("sh: {e}"))
}

/// Removes what a previous update left beside the executable (Windows can't delete a running
/// file, so the old one is only moved aside).
pub fn cleanup() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = fs::remove_file(sibling(&exe, ".old"));
        let _ = fs::remove_file(sibling(&exe, ".new"));
    }
}

/// Opens `url` in the default browser.
pub fn open_url(url: &str) {
    if !url.starts_with("https://github.com/") {
        return;
    }
    let mut cmd;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd = Command::new("rundll32");
        cmd.args(["url.dll,FileProtocolHandler", url]).creation_flags(0x0800_0000);
    }
    #[cfg(target_os = "macos")]
    {
        cmd = Command::new("open");
        cmd.arg(url);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        cmd = Command::new("xdg-open");
        cmd.arg(url);
    }
    let _ = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_numerically() {
        assert!(is_newer("v1.0.1", "1.0.0"));
        assert!(is_newer("1.10.0", "1.9.9"));
        assert!(is_newer("2.0", "1.9.9"));
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(!is_newer("v0.9.0", "1.0.0"));
        assert!(!is_newer("garbage", "1.0.0"));
    }

    #[test]
    fn a_pre_release_suffix_is_ignored() {
        assert_eq!(parse_version("v1.2.3-beta.1"), Some([1, 2, 3]));
        assert_eq!(parse_version("1.2"), Some([1, 2, 0]));
        assert_eq!(parse_version(""), None);
    }

    fn release() -> Release {
        let json = serde_json::json!({
            "tag_name": "v1.2.0",
            "assets": [
                {"name": "mosaix-windows-x64-setup.exe", "browser_download_url": "https://github.com/adamhlt/Mosaix/releases/download/v1.2.0/mosaix-windows-x64-setup.exe", "size": 10, "digest": "sha256:ABCD"},
                {"name": "mosaix-windows-x64-portable.zip", "browser_download_url": "u2", "size": 5},
                {"name": "mosaix-linux-x64.tar.gz", "browser_download_url": "u3", "size": 6},
                {"name": "mosaix-macos-universal.dmg", "browser_download_url": "u4", "size": 7},
            ]
        });
        parse_release(&json).unwrap()
    }

    #[test]
    fn a_release_is_read_from_the_api() {
        let r = release();
        assert_eq!(r.version, "1.2.0");
        assert_eq!(r.assets.len(), 4);
        assert_eq!(r.assets[0].sha256.as_deref(), Some("abcd"));
        assert_eq!(r.assets[1].sha256, None);
        assert_eq!(r.page(), "https://github.com/adamhlt/Mosaix/releases/tag/v1.2.0");
    }

    #[test]
    fn each_install_picks_its_own_file() {
        let r = release();
        let name = |m| pick_asset(m, &r.assets).map(|a| a.name);
        assert_eq!(name(Method::Installer).as_deref(), Some("mosaix-windows-x64-setup.exe"));
        assert_eq!(name(Method::Portable).as_deref(), Some("mosaix-windows-x64-portable.zip"));
        assert_eq!(name(Method::Binary).as_deref(), Some("mosaix-linux-x64.tar.gz"));
        assert_eq!(name(Method::MacApp).as_deref(), Some("mosaix-macos-universal.dmg"));
        assert_eq!(name(Method::Manual), None);
    }

    #[test]
    fn the_page_url_cannot_leave_the_repository() {
        let r = Release { tag: "v1/../../evil?x=1".into(), version: "1".into(), assets: Vec::new() };
        assert_eq!(r.page(), "https://github.com/adamhlt/Mosaix/releases/tag/v1....evilx1");
    }

    #[test]
    fn the_app_bundle_is_found_from_the_executable() {
        let exe = Path::new("/Applications/Mosaix.app/Contents/MacOS/mosaix");
        assert_eq!(mac_bundle(exe), Some(PathBuf::from("/Applications/Mosaix.app")));
        assert_eq!(mac_bundle(Path::new("/usr/local/bin/mosaix")), None);
        assert_eq!(detect_method(exe, "macos", "aarch64"), Method::MacApp);
        assert_eq!(detect_method(Path::new("/opt/mosaix/mosaix"), "macos", "aarch64"), Method::Manual);
    }

    #[test]
    fn unsupported_platforms_fall_back_to_the_release_page() {
        let exe = Path::new("/opt/mosaix/mosaix");
        assert_eq!(detect_method(exe, "linux", "x86_64"), Method::Binary);
        assert_eq!(detect_method(exe, "linux", "aarch64"), Method::Manual);
        assert_eq!(detect_method(exe, "freebsd", "x86_64"), Method::Manual);
        assert_eq!(detect_method(Path::new("C:/Apps/mosaix.exe"), "windows", "aarch64"), Method::Manual);
    }

    #[test]
    fn a_download_counts_its_bytes_and_wakes_the_window_as_it_goes() {
        // A body that arrives 10 KB at a time.
        struct Slow(Vec<u8>, usize);
        impl Read for Slow {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let n = (self.0.len() - self.1).min(10_000).min(buf.len());
                buf[..n].copy_from_slice(&self.0[self.1..self.1 + n]);
                self.1 += n;
                Ok(n)
            }
        }
        let data: Vec<u8> = (0..500_000u32).map(|i| (i % 251) as u8).collect();
        let done = AtomicU64::new(0);
        let wakes = std::cell::Cell::new(0);
        let seen = std::cell::RefCell::new(Vec::new());
        let mut out = Vec::new();
        let (total, hash) = copy_counting(&mut Slow(data.clone(), 0), &mut out, data.len() as u64, &done, &|| {
            wakes.set(wakes.get() + 1);
            seen.borrow_mut().push(done.load(Ordering::Relaxed));
        })
        .unwrap();
        assert_eq!((total, out.len()), (500_000, 500_000));
        assert_eq!(hash, Sha256::digest(&data).to_vec());
        // 50 reads of 1 % each: one wake per read, each with more bytes than the last.
        assert_eq!(wakes.get(), 50);
        assert!(seen.borrow().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn powershell_strings_are_quoted() {
        assert_eq!(ps_quote(Path::new("C:/it's/mosaix.exe")), "'C:/it''s/mosaix.exe'");
    }
}
