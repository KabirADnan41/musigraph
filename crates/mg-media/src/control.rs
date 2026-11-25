//! Playback control of the selected media session (SMTC), plus capability flags.
//!
//! Trust boundary: the session and everything it reports are UNTRUSTED. A command is only sent when the
//! matching capability flag is true; the backend re-checks the flags and the seek range on the freshly
//! resolved session (never a cached one). Failures carry a class only, never titles or app text.
//!
//! Invariants: at most ONE command in flight (a worker thread per command; each WinRT wait is bounded to
//! 3 s by `wait!`). Commands arriving while one runs, or within `MIN_GAP` of the last accepted one, are
//! dropped with `Dispatch::Busy`. `send` never blocks the caller. No queue exists. The real WinRT calls sit
//! behind `SessionBackend` so tests use a fake and never touch a live player.

use crate::PlaybackState;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Commands inside this window of the previous accepted command are dropped (key-repeat / double-tap guard).
pub const MIN_GAP: Duration = Duration::from_millis(150);

/// What the selected session reports it can do. Unknown or missing means `false` ("unavailable").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub can_play: bool,
    pub can_pause: bool,
    pub can_toggle: bool,
    pub can_next: bool,
    pub can_previous: bool,
    pub can_seek: bool,
}

/// Raw `PlaybackControls` booleans in the order play, pause, toggle, next, previous, seek;
/// `None` = the property could not be read.
pub type RawControls = [Option<bool>; 6];

impl Capabilities {
    pub fn from_raw(r: RawControls) -> Self {
        let f = |i: usize| r[i].unwrap_or(false);
        Self {
            can_play: f(0),
            can_pause: f(1),
            can_toggle: f(2),
            can_next: f(3),
            can_previous: f(4),
            can_seek: f(5),
        }
    }

    pub const fn all() -> Self {
        Self {
            can_play: true,
            can_pause: true,
            can_toggle: true,
            can_next: true,
            can_previous: true,
            can_seek: true,
        }
    }

    pub fn allows(&self, c: MediaCommand) -> bool {
        match c {
            MediaCommand::Play => self.can_play,
            MediaCommand::Pause => self.can_pause,
            MediaCommand::TogglePlayPause => self.can_toggle,
            MediaCommand::Next => self.can_next,
            MediaCommand::Previous => self.can_previous,
            MediaCommand::SeekTo(_) => self.can_seek,
        }
    }
}

impl Capabilities {
    /// The command to actually send, or `None` if unsupported. Toggle falls back to Play/Pause by the
    /// current playback `state` when the player has no toggle flag.
    pub fn resolve(&self, c: MediaCommand, state: PlaybackState) -> Option<MediaCommand> {
        if self.allows(c) {
            return Some(c);
        }
        match (c, state) {
            (MediaCommand::TogglePlayPause, PlaybackState::Playing) if self.can_pause => {
                Some(MediaCommand::Pause)
            }
            (MediaCommand::TogglePlayPause, PlaybackState::Paused | PlaybackState::Stopped)
                if self.can_play =>
            {
                Some(MediaCommand::Play)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MediaCommand {
    Play,
    Pause,
    TogglePlayPause,
    Next,
    Previous,
    SeekTo(f64),
}

/// Immediate answer of `send`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Started,
    Unsupported,
    Busy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailClass {
    /// The player answered "not done".
    Rejected,
    /// WinRT error or timeout.
    Error,
}

/// Outcome of the most recent command (pollable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandResult {
    Ok,
    Failed(FailClass),
    NoSession,
    Unsupported,
}

/// The only place real session calls happen. `execute` runs on a worker thread, must resolve the selected
/// session itself, re-check capabilities/seek range, and be bounded in time.
pub trait SessionBackend: Send + Sync + 'static {
    fn execute(&self, cmd: MediaCommand) -> CommandResult;
}

/// Seekable range in ticks: the session's Min/MaxSeekTime when both are readable and non-empty, else
/// Start/End.
pub fn seek_range(min: Option<i64>, max: Option<i64>, start: i64, end: i64) -> (i64, i64) {
    match (min, max) {
        (Some(a), Some(b)) if b > a => (a, b),
        _ => (start, end),
    }
}

/// Target for a seek in 100 ns ticks within `[start, end]`. NaN or an empty range => `None`; negative or huge
/// targets clamp to the range; arithmetic saturates.
pub fn seek_ticks(target_s: f64, start: i64, end: i64) -> Option<i64> {
    if target_s.is_nan() || end <= start {
        return None;
    }
    let rel = (target_s * 1e7).clamp(0.0, end.saturating_sub(start) as f64); // `as i64` saturates
    Some(start.saturating_add(rel as i64).min(end))
}

#[derive(Default)]
struct Inner {
    busy: AtomicBool,
    last_accept: Mutex<Option<Instant>>,
    /// Last issued command sequence number.
    seq: AtomicU64,
    /// Result with the sequence of the command that produced it; only a higher sequence replaces it.
    result: Mutex<Option<(u64, CommandResult)>>,
}

impl Inner {
    fn record(&self, seq: u64, r: CommandResult) {
        if let Ok(mut g) = self.result.lock() {
            if g.is_none_or(|(s, _)| seq > s) {
                *g = Some((seq, r));
            }
        }
    }
}

/// Sends commands to the selected session off the caller's thread.
pub struct MediaController {
    backend: Arc<dyn SessionBackend>,
    inner: Arc<Inner>,
}

/// Records the result and clears `busy` even if the backend panics.
struct Finish(Arc<Inner>, u64, Option<CommandResult>);
impl Drop for Finish {
    fn drop(&mut self) {
        let r = self
            .2
            .take()
            .unwrap_or(CommandResult::Failed(FailClass::Error));
        self.0.record(self.1, r);
        self.0.busy.store(false, Ordering::Release);
    }
}

impl MediaController {
    pub fn new(backend: Arc<dyn SessionBackend>) -> Self {
        Self {
            backend,
            inner: Arc::default(),
        }
    }

    /// Controller for the live Windows media sessions.
    #[cfg(windows)]
    pub fn system() -> Self {
        Self::new(Arc::new(crate::session::SystemBackend))
    }

    fn next_seq(&self) -> u64 {
        self.inner.seq.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// `caps` and `state` are from the latest snapshot; the backend re-checks on the fresh session.
    pub fn send(
        &self,
        cmd: MediaCommand,
        caps: &Capabilities,
        state: PlaybackState,
        now: Instant,
    ) -> Dispatch {
        let Some(cmd) = caps.resolve(cmd, state) else {
            let seq = self.next_seq();
            self.inner.record(seq, CommandResult::Unsupported);
            return Dispatch::Unsupported;
        };
        // Busy is only the dispatch answer: it records nothing and never touches the in-flight result.
        if self
            .inner
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Dispatch::Busy;
        }
        let gap_ok = self.inner.last_accept.lock().is_ok_and(|mut g| {
            let ok = g.is_none_or(|t| now.saturating_duration_since(t) >= MIN_GAP);
            if ok {
                *g = Some(now);
            }
            ok
        });
        if !gap_ok {
            self.inner.busy.store(false, Ordering::Release);
            return Dispatch::Busy;
        }
        let seq = self.next_seq();
        let (backend, inner) = (self.backend.clone(), self.inner.clone());
        let spawned = std::thread::Builder::new()
            .name("mg-media-cmd".into())
            .spawn(move || {
                let mut fin = Finish(inner, seq, None);
                fin.2 = Some(backend.execute(cmd));
            });
        if spawned.is_err() {
            // Thread could not start: result is Failed, the dispatch answer is Busy, and this attempt
            // already consumed the 150 ms gap.
            self.inner
                .record(seq, CommandResult::Failed(FailClass::Error));
            self.inner.busy.store(false, Ordering::Release);
            return Dispatch::Busy;
        }
        Dispatch::Started
    }

    pub fn is_busy(&self) -> bool {
        self.inner.busy.load(Ordering::Acquire)
    }

    /// Result of the most recent finished (or refused-as-unsupported) command.
    pub fn last_result(&self) -> Option<CommandResult> {
        self.inner
            .result
            .lock()
            .ok()
            .and_then(|g| g.map(|(_, r)| r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    const ST: PlaybackState = PlaybackState::Playing;

    struct Fake {
        calls: AtomicUsize,
        last: Mutex<Option<MediaCommand>>,
        reply: CommandResult,
        gate: Mutex<Option<mpsc::Receiver<()>>>,
    }
    impl Fake {
        fn new(reply: CommandResult) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                last: Mutex::new(None),
                reply,
                gate: Mutex::new(None),
            })
        }
    }
    impl SessionBackend for Fake {
        fn execute(&self, c: MediaCommand) -> CommandResult {
            *self.last.lock().unwrap() = Some(c);
            self.calls.fetch_add(1, Ordering::SeqCst);
            let gate = self.gate.lock().unwrap().take();
            if let Some(rx) = gate {
                let _ = rx.recv_timeout(Duration::from_secs(5));
            }
            self.reply
        }
    }
    fn settle(c: &MediaController) {
        let end = Instant::now() + Duration::from_secs(5);
        while c.is_busy() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(!c.is_busy());
    }

    #[test]
    fn capability_mapping_unknown_is_false() {
        assert_eq!(Capabilities::from_raw([None; 6]), Capabilities::default());
        let c = Capabilities::from_raw([
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            Some(true),
        ]);
        assert!(c.can_play && !c.can_pause && !c.can_toggle);
        assert!(c.can_next && !c.can_previous && c.can_seek);
        assert!(Capabilities::all().allows(MediaCommand::SeekTo(1.0)));
        assert!(!c.allows(MediaCommand::Pause) && c.allows(MediaCommand::Next));
    }

    #[test]
    fn unsupported_never_reaches_backend() {
        let f = Fake::new(CommandResult::Ok);
        let c = MediaController::new(f.clone());
        let caps = Capabilities {
            can_next: true,
            ..Default::default()
        };
        let t = Instant::now();
        assert_eq!(
            c.send(MediaCommand::Pause, &caps, ST, t),
            Dispatch::Unsupported
        );
        assert_eq!(c.last_result(), Some(CommandResult::Unsupported));
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        assert_eq!(c.send(MediaCommand::Next, &caps, ST, t), Dispatch::Started);
        settle(&c);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(c.last_result(), Some(CommandResult::Ok));
    }

    #[test]
    fn failure_and_no_session_are_reported() {
        for r in [
            CommandResult::Failed(FailClass::Rejected),
            CommandResult::NoSession,
        ] {
            let c = MediaController::new(Fake::new(r));
            let d = c.send(MediaCommand::Play, &Capabilities::all(), ST, Instant::now());
            assert_eq!(d, Dispatch::Started);
            settle(&c);
            assert_eq!(c.last_result(), Some(r));
        }
    }

    #[test]
    fn one_in_flight_and_gap_window() {
        let f = Fake::new(CommandResult::Ok);
        let (tx, rx) = mpsc::channel();
        *f.gate.lock().unwrap() = Some(rx);
        let c = MediaController::new(f.clone());
        let t0 = Instant::now();
        let all = Capabilities::all();
        let at = |ms| t0 + Duration::from_millis(ms);
        assert_eq!(c.send(MediaCommand::Next, &all, ST, t0), Dispatch::Started);
        // in flight: even long after the gap a second command is Busy and never reaches the backend
        assert_eq!(
            c.send(MediaCommand::Next, &all, ST, at(1000)),
            Dispatch::Busy
        );
        tx.send(()).unwrap();
        settle(&c);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        // finished but inside the 150 ms window: dropped; after it: accepted
        assert_eq!(
            c.send(MediaCommand::Next, &all, ST, at(100)),
            Dispatch::Busy
        );
        assert_eq!(
            c.send(MediaCommand::Next, &all, ST, at(160)),
            Dispatch::Started
        );
        settle(&c);
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    }

    fn toggle_via(caps: Capabilities, st: PlaybackState) -> (Dispatch, Option<MediaCommand>) {
        let f = Fake::new(CommandResult::Ok);
        let c = MediaController::new(f.clone());
        let d = c.send(MediaCommand::TogglePlayPause, &caps, st, Instant::now());
        if d == Dispatch::Started {
            settle(&c);
        }
        let last = *f.last.lock().unwrap();
        (d, last)
    }

    #[test]
    fn toggle_falls_back_to_play_or_pause_by_state() {
        use PlaybackState::*;
        let tog = Capabilities {
            can_toggle: true,
            ..Default::default()
        };
        let pp = Capabilities {
            can_play: true,
            can_pause: true,
            ..Default::default()
        };
        let started = |m| (Dispatch::Started, Some(m));
        assert_eq!(
            toggle_via(tog, Playing),
            started(MediaCommand::TogglePlayPause)
        );
        assert_eq!(toggle_via(pp, Playing), started(MediaCommand::Pause));
        assert_eq!(toggle_via(pp, Paused), started(MediaCommand::Play));
        assert_eq!(toggle_via(pp, Stopped), started(MediaCommand::Play));
        assert_eq!(toggle_via(pp, Other), (Dispatch::Unsupported, None));
        let only_play = Capabilities {
            can_play: true,
            ..Default::default()
        };
        assert_eq!(
            toggle_via(only_play, Playing),
            (Dispatch::Unsupported, None)
        );
        assert_eq!(
            toggle_via(Capabilities::default(), Paused),
            (Dispatch::Unsupported, None)
        );
    }

    #[test]
    fn older_finish_never_overwrites_newer_result() {
        let f = Fake::new(CommandResult::Ok);
        let (tx, rx) = mpsc::channel();
        *f.gate.lock().unwrap() = Some(rx);
        let c = MediaController::new(f.clone());
        let caps = Capabilities {
            can_next: true,
            ..Default::default()
        };
        let t = Instant::now();
        assert_eq!(c.send(MediaCommand::Next, &caps, ST, t), Dispatch::Started);
        // a later unsupported command is recorded while the first is in flight
        assert_eq!(
            c.send(MediaCommand::Pause, &caps, ST, t),
            Dispatch::Unsupported
        );
        // Busy records nothing
        assert_eq!(c.send(MediaCommand::Next, &caps, ST, t), Dispatch::Busy);
        assert_eq!(c.last_result(), Some(CommandResult::Unsupported));
        tx.send(()).unwrap();
        settle(&c);
        assert_eq!(c.last_result(), Some(CommandResult::Unsupported));
    }

    #[test]
    fn seek_range_prefers_min_max() {
        assert_eq!(seek_range(Some(5), Some(50), 0, 100), (5, 50));
        assert_eq!(seek_range(None, Some(50), 0, 100), (0, 100));
        assert_eq!(seek_range(Some(5), None, 0, 100), (0, 100));
        assert_eq!(seek_range(Some(9), Some(9), 0, 100), (0, 100));
    }

    #[test]
    fn backend_panic_frees_the_controller() {
        struct Boom;
        impl SessionBackend for Boom {
            fn execute(&self, _: MediaCommand) -> CommandResult {
                panic!("boom")
            }
        }
        let c = MediaController::new(Arc::new(Boom));
        let d = c.send(MediaCommand::Play, &Capabilities::all(), ST, Instant::now());
        assert_eq!(d, Dispatch::Started);
        settle(&c);
        assert_eq!(
            c.last_result(),
            Some(CommandResult::Failed(FailClass::Error))
        );
    }

    #[test]
    fn seek_ticks_clamps_and_saturates() {
        let s = 10_000_000i64;
        assert_eq!(seek_ticks(10.0, 0, 100 * s), Some(10 * s));
        assert_eq!(seek_ticks(-5.0, 0, 100 * s), Some(0));
        assert_eq!(seek_ticks(1e30, 0, 100 * s), Some(100 * s));
        assert_eq!(seek_ticks(f64::INFINITY, 0, 100 * s), Some(100 * s));
        assert_eq!(seek_ticks(f64::NEG_INFINITY, 0, 100 * s), Some(0));
        assert_eq!(seek_ticks(f64::NAN, 0, 100 * s), None);
        assert_eq!(seek_ticks(5.0, 7, 7), None);
        assert_eq!(seek_ticks(5.0, 7, 3), None);
        assert_eq!(seek_ticks(1.0, 5 * s, 10 * s), Some(6 * s));
        assert_eq!(seek_ticks(1e30, 0, i64::MAX), Some(i64::MAX));
        // hostile range: must not panic (saturating arithmetic)
        assert!(seek_ticks(1e30, i64::MIN, i64::MAX).is_some());
        assert_eq!(seek_ticks(1e30, i64::MAX - 5, i64::MAX), Some(i64::MAX));
    }
}
