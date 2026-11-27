//! Windows-only watcher: polls the system media sessions on its own thread (MTA) and publishes the
//! selected session as a `NowPlaying` snapshot. WinRT calls through windows-rs are safe code.

use crate::control::{
    Capabilities, CommandResult, FailClass, MediaCommand, SessionBackend, seek_target,
};
use crate::{
    Candidate, MAX_ART_BYTES, NowPlaying, PlaybackState, sanitize_text, select_session,
    timeline_seconds,
};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use windows::Media::Control::{
    GlobalSystemMediaTransportControlsSession as Session,
    GlobalSystemMediaTransportControlsSessionManager as Manager,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus as Status,
};
use windows::Storage::Streams::DataReader;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};

const POLL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct Shared {
    now: Option<NowPlaying>,
    last_error: Option<String>,
    polls: u64,
}

/// Handle to the watcher thread; dropping it stops and joins the thread (bounded).
pub struct NowPlayingWatcher {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    shared: Arc<Mutex<Shared>>,
    thread: Option<JoinHandle<()>>,
}

fn state_of(s: Status) -> PlaybackState {
    if s == Status::Playing {
        PlaybackState::Playing
    } else if s == Status::Paused {
        PlaybackState::Paused
    } else if s == Status::Stopped {
        PlaybackState::Stopped
    } else {
        PlaybackState::Other
    }
}

fn read_art(
    props: &windows::Media::Control::GlobalSystemMediaTransportControlsSessionMediaProperties,
) -> Option<Arc<[u8]>> {
    let stream = wait!(props.Thumbnail().ok()?.OpenReadAsync())?;
    let size = stream.Size().ok()?;
    if size == 0 || size > MAX_ART_BYTES as u64 {
        return None; // absent or implausibly large: ignore rather than allocate
    }
    let reader = DataReader::CreateDataReader(&stream).ok()?;
    let n = wait!(reader.LoadAsync(size as u32))?.min(size as u32);
    if n == 0 {
        return None;
    }
    let mut buf = vec![0u8; n as usize];
    reader.ReadBytes(&mut buf).ok()?;
    Some(Arc::from(buf))
}

fn track_hash(title: &str, artist: &str, album: &str, app: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (title, artist, album, app).hash(&mut h);
    h.finish()
}

/// Current sessions with their selection candidates (resolved fresh on every call).
fn candidates(mgr: &Manager) -> Result<(Vec<Session>, Vec<Candidate>), String> {
    let sessions: Vec<Session> = mgr
        .GetSessions()
        .map_err(|x| format!("GetSessions: {x}"))?
        .into_iter()
        .collect();
    let current_app = mgr
        .GetCurrentSession()
        .ok()
        .and_then(|s| s.SourceAppUserModelId().ok());
    let cands: Vec<Candidate> = sessions
        .iter()
        .enumerate()
        .map(|(index, s)| Candidate {
            index,
            state: s
                .GetPlaybackInfo()
                .and_then(|i| i.PlaybackStatus())
                .map_or(PlaybackState::Other, state_of),
            is_system_current: current_app
                .as_ref()
                .is_some_and(|c| s.SourceAppUserModelId().is_ok_and(|a| a == *c)),
        })
        .collect();
    Ok((sessions, cands))
}

/// Capability flags from the session's `PlaybackInfo().Controls()`; anything unreadable is `false`.
fn read_caps(s: &Session) -> Capabilities {
    let Ok(c) = s.GetPlaybackInfo().and_then(|i| i.Controls()) else {
        return Capabilities::default();
    };
    Capabilities::from_raw([
        c.IsPlayEnabled().ok(),
        c.IsPauseEnabled().ok(),
        c.IsPlayPauseToggleEnabled().ok(),
        c.IsNextEnabled().ok(),
        c.IsPreviousEnabled().ok(),
        c.IsPlaybackPositionEnabled().ok(),
    ])
}

/// One poll: returns the selected session's snapshot (reusing `prev` artwork when the track is unchanged).
fn poll(
    mgr: &Manager,
    prev: Option<&NowPlaying>,
    retry_art: bool,
) -> Result<Option<NowPlaying>, String> {
    let (sessions, cands) = candidates(mgr)?;
    let Some(i) = select_session(&cands) else {
        return Ok(None);
    };
    let s = &sessions[i];
    let props = wait!(s.TryGetMediaPropertiesAsync())
        .ok_or("media properties unavailable (timeout or error)")?;
    let field = |r: windows::core::Result<windows::core::HSTRING>| {
        sanitize_text(&r.map(|h| h.to_string_lossy()).unwrap_or_default())
    };
    let (title, artist, album) = (
        field(props.Title()),
        field(props.Artist()),
        field(props.AlbumTitle()),
    );
    let app_id = field(s.SourceAppUserModelId());
    let id = track_hash(&title, &artist, &album, &app_id);

    // Timeline: position at the session's last update, converted to "seconds at the time we sampled".
    let (mut position_s, mut duration_s) = (None, None);
    if let Ok(t) = s.GetTimelineProperties() {
        let tick =
            |r: windows::core::Result<windows::Foundation::TimeSpan>| r.map_or(0, |d| d.Duration);
        let now_unix_ticks = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_nanos() / 100).unwrap_or(i64::MAX));
        let updated = t.LastUpdatedTime().map_or(0, |d| d.UniversalTime);
        let playing = cands[i].state == PlaybackState::Playing;
        if let Some((p, d)) = timeline_seconds(
            tick(t.StartTime()),
            tick(t.EndTime()),
            tick(t.Position()),
            updated,
            now_unix_ticks,
            playing,
        ) {
            (position_s, duration_s) = (Some(p), Some(d));
        }
    }
    let sampled_at = Instant::now();
    let same = prev.is_some_and(|p| p.track_id == id);
    let art = if same && !(retry_art && prev.is_some_and(|p| p.art.is_none())) {
        prev.and_then(|p| p.art.clone())
    } else {
        read_art(&props)
    };
    Ok(Some(NowPlaying {
        title,
        artist,
        album,
        app_id,
        state: cands[i].state,
        position_s,
        duration_s,
        sampled_at,
        art,
        track_id: id,
        caps: read_caps(s),
    }))
}

/// Real WinRT backend: resolves the selected session at send time and performs the call (bounded waits).
pub(crate) struct SystemBackend;

impl SessionBackend for SystemBackend {
    fn execute(&self, cmd: MediaCommand) -> CommandResult {
        // SAFETY: balanced by CoUninitialize below on this thread; MTA so blocking waits are allowed.
        #[allow(unsafe_code)]
        let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if com.is_err() {
            return CommandResult::Failed(FailClass::Error);
        }
        let r = run_command(cmd);
        // SAFETY: paired with the successful CoInitializeEx above; WinRT objects dropped in run_command.
        #[allow(unsafe_code)]
        unsafe {
            CoUninitialize()
        };
        r
    }
}

fn run_command(cmd: MediaCommand) -> CommandResult {
    let Some(mgr) = wait!(Manager::RequestAsync()) else {
        return CommandResult::Failed(FailClass::Error); // failure or timeout, not "no session"
    };
    let Ok((sessions, cands)) = candidates(&mgr) else {
        return CommandResult::Failed(FailClass::Error);
    };
    let Some(i) = select_session(&cands) else {
        return CommandResult::NoSession;
    };
    let s = &sessions[i];
    // Fresh caps and state: Toggle may translate to Play/Pause when the player has no toggle flag.
    let Some(cmd) = read_caps(s).resolve(cmd, cands[i].state) else {
        return CommandResult::Unsupported;
    };
    let done = match cmd {
        MediaCommand::Play => wait!(s.TryPlayAsync()),
        MediaCommand::Pause => wait!(s.TryPauseAsync()),
        MediaCommand::TogglePlayPause => wait!(s.TryTogglePlayPauseAsync()),
        MediaCommand::Next => wait!(s.TrySkipNextAsync()),
        MediaCommand::Previous => wait!(s.TrySkipPreviousAsync()),
        MediaCommand::SeekTo(secs) => {
            let Ok(t) = s.GetTimelineProperties() else {
                return CommandResult::Failed(FailClass::Error);
            };
            let tick = |r: windows::core::Result<windows::Foundation::TimeSpan>| {
                r.map_or(0, |d| d.Duration)
            };
            let opt = |r: windows::core::Result<windows::Foundation::TimeSpan>| {
                r.ok().map(|d| d.Duration)
            };
            let Some(target) = seek_target(
                secs,
                tick(t.StartTime()),
                tick(t.EndTime()),
                opt(t.MinSeekTime()),
                opt(t.MaxSeekTime()),
            ) else {
                return CommandResult::Unsupported;
            };
            wait!(s.TryChangePlaybackPositionAsync(target))
        }
    };
    match done {
        Some(true) => CommandResult::Ok,
        Some(false) => CommandResult::Failed(FailClass::Rejected),
        None => CommandResult::Failed(FailClass::Error),
    }
}

fn run(stop: &AtomicBool, shared: &Mutex<Shared>) {
    // SAFETY: balanced by CoUninitialize below on this thread; MTA so blocking `.join()` is allowed.
    #[allow(unsafe_code)]
    let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if com.is_err() {
        if let Ok(mut s) = shared.lock() {
            s.last_error = Some(format!("CoInitializeEx failed: {com:?}"));
        }
        return;
    }
    let mgr = wait!(Manager::RequestAsync());
    match mgr {
        None => {
            if let Ok(mut s) = shared.lock() {
                s.last_error = Some("media session manager unavailable".to_string());
            }
        }
        Some(mgr) => {
            while !stop.load(Ordering::Acquire) {
                let prev = shared.lock().ok().and_then(|s| s.now.clone());
                let retry = shared.lock().is_ok_and(|s| s.polls % 10 == 0);
                let r = poll(&mgr, prev.as_ref(), retry);
                if let Ok(mut s) = shared.lock() {
                    s.polls += 1;
                    match r {
                        Ok(v) => {
                            s.now = v;
                            s.last_error = None;
                        }
                        Err(e) => s.last_error = Some(e),
                    }
                }
                let end = Instant::now() + POLL;
                while Instant::now() < end && !stop.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
        }
    }
    // SAFETY: paired with the successful CoInitializeEx above; all WinRT objects were dropped with `mgr`.
    #[allow(unsafe_code)]
    unsafe {
        CoUninitialize()
    };
}

impl NowPlayingWatcher {
    pub fn start() -> Result<Self, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (s2, d2, sh2) = (stop.clone(), done.clone(), shared.clone());
        let thread = std::thread::Builder::new()
            .name("mg-media-watcher".into())
            .spawn(move || {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&s2, &sh2)))
                    .is_err()
                {
                    if let Ok(mut s) = sh2.lock() {
                        s.last_error = Some("media watcher panicked and stopped".into());
                    }
                }
                d2.store(true, Ordering::Release);
            })
            .map_err(|e| format!("cannot start media watcher: {e}"))?;
        Ok(Self {
            stop,
            done,
            shared,
            thread: Some(thread),
        })
    }

    /// Latest snapshot of the selected session; `None` when no usable session exists. A snapshot older
    /// than `STALE_AFTER` (watcher stalled) is reported as `None` so stale song info is never shown.
    pub fn now_playing(&self) -> Option<NowPlaying> {
        let n = self.shared.lock().ok()?.now.clone()?;
        (Instant::now().saturating_duration_since(n.sampled_at) < crate::STALE_AFTER
            && n.is_usable())
        .then_some(n)
    }

    pub fn last_error(&self) -> Option<String> {
        self.shared.lock().ok()?.last_error.clone()
    }

    pub fn polls(&self) -> u64 {
        self.shared.lock().map_or(0, |s| s.polls)
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let Some(t) = self.thread.take() else { return };
        let deadline = Instant::now() + Duration::from_secs(3);
        while !self.done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if self.done.load(Ordering::Acquire) {
            let _ = t.join();
        } // else detached: a stalled WinRT call must never hang shutdown
    }
}

impl Drop for NowPlayingWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}
