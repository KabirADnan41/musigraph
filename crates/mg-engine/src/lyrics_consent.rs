//! First-use consent for online lyrics (LRCLIB). Pure + tiny persistence.
//!
//! Persists ONLY the tri-state (`%APPDATA%\Musigraph\lyrics-consent.txt`), never track data.
//! Missing/corrupt/oversize/unknown content is `Unasked`, which never allows network access.
//! `Granted` is produced only by `grant()` (an explicit user action), never by `load`.

use std::io;
use std::path::{Path, PathBuf};

/// Owner-approved prompt text, verbatim.
pub const PROMPT: &str = "Enable automatic online lyrics?\nMusigraph can retrieve lyrics from LRCLIB using the currently playing track's title, artist, album and duration. These details, your connection IP address and the Musigraph application identifier will be visible to the service.\nNo account is required. Musigraph does not upload your audio or maintain a listening history. LRCLIB's data-retention practices have not been verified.\nYou can disable online lyrics at any time.";

const FILE: &str = "lyrics-consent.txt";
const MAX_LEN: u64 = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Consent {
    #[default]
    Unasked,
    Granted,
    Declined,
}

impl Consent {
    /// True only for an explicit grant.
    pub fn allows_network(self) -> bool {
        self == Consent::Granted
    }

    fn token(self) -> &'static str {
        match self {
            Consent::Unasked => "unasked",
            Consent::Granted => "granted",
            Consent::Declined => "declined",
        }
    }

    /// Persist the explicit "Enable" choice and return the new state.
    pub fn grant() -> Consent {
        Self::grant_to(app_dir().as_deref())
    }

    /// Persist "Disable"/"No" (also the revoke path) and return the new state.
    pub fn decline() -> (Consent, bool) {
        Self::decline_to(app_dir().as_deref())
    }

    pub fn grant_to(dir: Option<&Path>) -> Consent {
        // A failed save still returns the in-memory choice for this session.
        if let Some(d) = dir {
            let _ = save_to(d, Consent::Granted);
        }
        Consent::Granted
    }

    /// Revokes consent: the session is Declined at once; the bool says whether that is durable on disk. If the
    /// write fails, the stored file is deleted instead so a stale "granted" can never return at the next start
    /// (fail closed: a missing file loads as Unasked). `false` only when both fail; the caller should warn.
    pub fn decline_to(dir: Option<&Path>) -> (Consent, bool) {
        let durable = dir.is_none_or(|d| {
            save_to(d, Consent::Declined).is_ok() || {
                let p = d.join(FILE);
                !p.exists() || std::fs::remove_file(p).is_ok()
            }
        });
        (Consent::Declined, durable)
    }
}

fn app_dir() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").filter(|v| !v.is_empty())?;
    Some(PathBuf::from(base).join("Musigraph"))
}

pub fn load() -> Consent {
    app_dir().map_or(Consent::Unasked, |d| load_from(&d))
}

pub fn save(c: Consent) -> io::Result<()> {
    match app_dir() {
        Some(d) => save_to(&d, c),
        None => Ok(()), // in-memory only
    }
}

pub fn load_from(dir: &Path) -> Consent {
    let path = dir.join(FILE);
    match std::fs::metadata(&path) {
        Ok(m) if m.is_file() && m.len() <= MAX_LEN => {}
        _ => return Consent::Unasked,
    }
    match std::fs::read_to_string(&path).as_deref().map(str::trim) {
        Ok("granted") => Consent::Granted,
        Ok("declined") => Consent::Declined,
        _ => Consent::Unasked,
    }
}

pub fn save_to(dir: &Path, c: Consent) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{FILE}.tmp"));
    let res = std::fs::write(&tmp, c.token()).and_then(|()| std::fs::rename(&tmp, dir.join(FILE)));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Warning shown when a revocation could not be written or deleted on disk.
pub const REVOKE_WARNING: &str = "Online lyrics are disabled for this session only: the revocation could not be saved. Consent may return at the next start unless the file lyrics-consent.txt is removed manually.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RevokeResult {
    pub durable: bool,
    pub warning: Option<&'static str>,
}

/// Runtime gate for online lyrics: persisted consent plus an in-memory session revocation that wins even
/// if persisting the revocation fails. The engine must consult `allows_network()` before every request and
/// mirror changes into `LrclibClient::set_consent`. Local lyrics never consult it.
#[derive(Clone, Copy, Debug)]
pub struct OnlineLyricsGate {
    consent: Consent,
    session_revoked: bool,
}

impl OnlineLyricsGate {
    /// Startup / restart: stored state only (a stored grant is honoured, nothing is asked).
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            consent: load_from(dir),
            session_revoked: false,
        }
    }

    pub fn consent(&self) -> Consent {
        self.consent
    }

    pub fn allows_network(&self) -> bool {
        self.consent.allows_network() && !self.session_revoked
    }

    /// Explicit user action only.
    pub fn grant(&mut self, dir: Option<&Path>) {
        self.consent = Consent::grant_to(dir);
        self.session_revoked = false;
    }

    /// Disables retrieval at once (in memory first), then persists; never claims durability it lacks.
    pub fn revoke(&mut self, dir: Option<&Path>) -> RevokeResult {
        self.session_revoked = true;
        self.consent = Consent::Declined;
        let (_, durable) = Consent::decline_to(dir);
        RevokeResult {
            durable,
            warning: (!durable).then_some(REVOKE_WARNING),
        }
    }
}

/// Only an online-sourced lyric load is gated; local lyrics are always allowed.
pub fn online_allowed_for_source(local: bool, gate: &OnlineLyricsGate) -> bool {
    local || gate.allows_network()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tdir(name: &str) -> PathBuf {
        let d =
            PathBuf::from(r"F:\musigraph\temp").join(format!("lc-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn default_and_missing_are_unasked() {
        assert_eq!(Consent::default(), Consent::Unasked);
        assert!(!Consent::Unasked.allows_network());
        assert!(!Consent::Declined.allows_network());
        assert!(Consent::Granted.allows_network());
        let d = tdir("missing");
        assert_eq!(load_from(&d), Consent::Unasked);
        assert_eq!(load_from(&d.join("nope")), Consent::Unasked);
    }

    #[test]
    fn corrupt_garbage_oversize_are_unasked() {
        let d = tdir("bad");
        let f = d.join(FILE);
        for bytes in [
            &b""[..],
            b"GRANTED?",
            b"yes",
            b"\xff\xfe\x00",
            b"granted granted granted granted granted",
        ] {
            std::fs::write(&f, bytes).unwrap();
            let c = load_from(&d);
            assert_eq!(c, Consent::Unasked, "{bytes:?}");
            assert!(!c.allows_network());
        }
        std::fs::write(&f, vec![b'a'; 1 << 20]).unwrap();
        assert_eq!(load_from(&d), Consent::Unasked);
        std::fs::remove_file(&f).unwrap();
        std::fs::create_dir(&f).unwrap(); // a directory in place of the file
        assert_eq!(load_from(&d), Consent::Unasked);
    }

    #[test]
    fn round_trip_and_no_partial_file() {
        let d = tdir("rt");
        for c in [Consent::Granted, Consent::Declined, Consent::Unasked] {
            save_to(&d, c).unwrap();
            assert_eq!(load_from(&d), c);
        }
        let names: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from(FILE)]);
    }

    #[test]
    fn grant_and_revoke_are_explicit() {
        let d = tdir("grant");
        assert_eq!(load_from(&d), Consent::Unasked);
        assert_eq!(Consent::grant_to(Some(&d)), Consent::Granted);
        assert_eq!(load_from(&d), Consent::Granted);
        assert_eq!(Consent::decline_to(Some(&d)), (Consent::Declined, true));
        assert_eq!(load_from(&d), Consent::Declined);
        assert_eq!(Consent::grant_to(None), Consent::Granted); // in-memory only
    }

    #[test]
    fn unicode_path() {
        let d = tdir("ünï-音楽").join("Müsigraph 日本");
        save_to(&d, Consent::Granted).unwrap();
        assert_eq!(load_from(&d), Consent::Granted);
    }

    #[test]
    fn unwritable_location_errors_without_panic() {
        let d = tdir("ro");
        let blocker = d.join("file");
        std::fs::write(&blocker, b"x").unwrap();
        assert!(save_to(&blocker.join("sub"), Consent::Granted).is_err());
        assert_eq!(
            Consent::grant_to(Some(&blocker.join("sub"))),
            Consent::Granted
        );
        assert_eq!(load_from(&blocker.join("sub")), Consent::Unasked);
    }

    #[test]
    fn prompt_is_verbatim() {
        assert_eq!(
            PROMPT,
            "Enable automatic online lyrics?\nMusigraph can retrieve lyrics from LRCLIB using the currently playing track's title, artist, album and duration. These details, your connection IP address and the Musigraph application identifier will be visible to the service.\nNo account is required. Musigraph does not upload your audio or maintain a listening history. LRCLIB's data-retention practices have not been verified.\nYou can disable online lyrics at any time."
        );
    }
}

#[cfg(test)]
mod revoke_tests {
    use super::*;

    #[test]
    fn a_failed_revoke_write_deletes_the_stored_grant_instead_of_leaving_it() {
        let d =
            PathBuf::from(r"F:\musigraph\temp").join(format!("lc-revoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        save_to(&d, Consent::Granted).unwrap();
        assert_eq!(load_from(&d), Consent::Granted);
        // make the temp-file write fail: a directory sits where the temp file must go
        std::fs::create_dir(d.join(format!("{FILE}.tmp"))).unwrap();
        let (c, durable) = Consent::decline_to(Some(&d));
        assert_eq!(c, Consent::Declined);
        assert!(durable);
        assert_eq!(
            load_from(&d),
            Consent::Unasked,
            "stale grant must not survive"
        );
        assert!(!load_from(&d).allows_network());
    }

    #[test]
    fn a_normal_revoke_persists_declined() {
        let d =
            PathBuf::from(r"F:\musigraph\temp").join(format!("lc-revoke2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        save_to(&d, Consent::Granted).unwrap();
        assert_eq!(Consent::decline_to(Some(&d)), (Consent::Declined, true));
        assert_eq!(load_from(&d), Consent::Declined);
        assert_eq!(Consent::decline_to(None), (Consent::Declined, true));
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use mg_media::lrclib::{HttpResponse, LrclibClient, Outcome, Query, Transport, TransportError};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const BODY: &str = r#"{"id":1,"trackName":"Synthetic Song","artistName":"Test Artist","albumName":"","duration":180,"instrumental":false,"plainLyrics":"la la la\nna na","syncedLyrics":"[00:01.00]la la la\n[00:03.50]na na"}"#;

    #[derive(Clone, Default)]
    struct Mock {
        calls: Arc<Mutex<u32>>,
        block: bool,
        saw_cancel: Arc<AtomicBool>,
        started: Arc<AtomicBool>,
    }
    impl Mock {
        fn n(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }
    impl Transport for Mock {
        fn get(
            &self,
            _: &str,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<HttpResponse, TransportError> {
            *self.calls.lock().unwrap() += 1;
            self.started.store(true, Ordering::Release);
            if self.block {
                let end = Instant::now() + Duration::from_secs(10);
                while Instant::now() < end {
                    if cancelled() {
                        self.saw_cancel.store(true, Ordering::Release);
                        return Err(TransportError::Cancelled);
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                return Err(TransportError::Timeout);
            }
            Ok(HttpResponse {
                status: 200,
                retry_after: None,
                body: BODY.as_bytes().to_vec(),
            })
        }
    }

    fn q(t: &str) -> Query {
        Query::new(t, "Test Artist", "", Some(180.0)).unwrap()
    }
    fn tdir(name: &str) -> PathBuf {
        let d =
            PathBuf::from(r"F:\musigraph\temp").join(format!("lc-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn wait_until(mut f: impl FnMut() -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs(3);
        while Instant::now() < end {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }
    fn client(m: &Mock) -> LrclibClient {
        let m2 = m.clone();
        LrclibClient::start(move || Some(m2)).unwrap()
    }
    /// One engine tick: the gate is mirrored into the client, as the engine must do.
    fn tick(c: &mut LrclibClient, g: &OnlineLyricsGate, t: Instant, q: &Query) -> Option<Outcome> {
        c.set_consent(g.allows_network());
        c.update(t, Some(q)).cloned()
    }

    #[test]
    fn full_lifecycle_grant_revoke_restart() {
        let d = tdir("gate");
        let m = Mock::default();
        let mut c = client(&m);
        let t0 = Instant::now();
        let song = q("Synthetic Song");
        let mut g = OnlineLyricsGate::from_dir(&d);
        // (a) unasked: nothing leaves
        assert_eq!(g.consent(), Consent::Unasked);
        for i in 0..200u64 {
            assert!(tick(&mut c, &g, t0 + Duration::from_millis(i * 50), &song).is_none());
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(m.n(), 0);
        // (b) grant -> one lookup -> lyrics
        g.grant(Some(&d));
        assert!(g.allows_network());
        let t1 = t0 + Duration::from_secs(20);
        tick(&mut c, &g, t1, &song);
        let t2 = t1 + Duration::from_secs(2);
        assert!(wait_until(|| matches!(
            tick(&mut c, &g, t2, &song),
            Some(Outcome::Synced(_))
        )));
        assert_eq!(m.n(), 1);
        // (c) revoke: immediate, lyrics cleared, no more calls even after a track change
        let r = g.revoke(Some(&d));
        assert_eq!(
            r,
            RevokeResult {
                durable: true,
                warning: None
            }
        );
        assert!(!g.allows_network());
        assert!(tick(&mut c, &g, t2, &song).is_none());
        for k in 0..30u64 {
            let other = q(&format!("Other {k}"));
            assert!(tick(&mut c, &g, t2 + Duration::from_secs(10 * (k + 1)), &other).is_none());
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(m.n(), 1);
        // (d) restart after durable revoke
        let g2 = OnlineLyricsGate::from_dir(&d);
        assert_eq!(g2.consent(), Consent::Declined);
        assert!(!g2.allows_network());
        let m3 = Mock::default();
        let mut c3 = client(&m3);
        for i in 0..100u64 {
            assert!(tick(&mut c3, &g2, t0 + Duration::from_millis(i * 100), &song).is_none());
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(m3.n(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn revoke_cancels_an_in_flight_request() {
        let d = tdir("gate-inflight");
        let m = Mock {
            block: true,
            ..Mock::default()
        };
        let mut c = client(&m);
        let mut g = OnlineLyricsGate::from_dir(&d);
        g.grant(Some(&d));
        let t0 = Instant::now();
        let song = q("Synthetic Song");
        tick(&mut c, &g, t0, &song);
        tick(&mut c, &g, t0 + Duration::from_secs(2), &song);
        assert!(wait_until(|| m.started.load(Ordering::Acquire)));
        g.revoke(Some(&d));
        assert!(tick(&mut c, &g, t0 + Duration::from_secs(3), &song).is_none());
        assert!(wait_until(|| m.saw_cancel.load(Ordering::Acquire)));
        assert_eq!(m.n(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn write_fails_but_delete_works_is_durable_and_restarts_unasked() {
        let d = tdir("gate-del");
        let mut g = OnlineLyricsGate::from_dir(&d);
        g.grant(Some(&d));
        std::fs::create_dir(d.join(format!("{FILE}.tmp"))).unwrap();
        let r = g.revoke(Some(&d));
        assert!(r.durable && r.warning.is_none());
        assert!(!g.allows_network());
        let g2 = OnlineLyricsGate::from_dir(&d);
        assert_eq!(g2.consent(), Consent::Unasked);
        let m = Mock::default();
        let mut c = client(&m);
        let t0 = Instant::now();
        for i in 0..50u64 {
            tick(
                &mut c,
                &g2,
                t0 + Duration::from_secs(i * 3),
                &q("Synthetic Song"),
            );
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(m.n(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn write_and_delete_both_fail_disables_this_session_and_warns() {
        let d = tdir("gate-fail");
        let m = Mock::default();
        let mut c = client(&m);
        let mut g = OnlineLyricsGate::from_dir(&d);
        g.grant(Some(&d));
        // the target path becomes a non-empty directory: rename onto it and remove_file both fail
        std::fs::remove_file(d.join(FILE)).unwrap();
        std::fs::create_dir(d.join(FILE)).unwrap();
        std::fs::write(d.join(FILE).join("x"), b"x").unwrap();
        let r = g.revoke(Some(&d));
        assert!(!r.durable);
        let w = r.warning.expect("warning");
        assert!(w.contains("this session only") && w.contains("could not be saved"));
        assert!(!w.contains("permanently") && !w.contains("has been saved"));
        assert!(!g.allows_network());
        let t0 = Instant::now();
        for i in 0..100u64 {
            assert!(
                tick(
                    &mut c,
                    &g,
                    t0 + Duration::from_secs(i * 3),
                    &q("Synthetic Song")
                )
                .is_none()
            );
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(m.n(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn gate_only_governs_online_sources() {
        let d = tdir("gate-local");
        let mut g = OnlineLyricsGate::from_dir(&d);
        assert!(online_allowed_for_source(true, &g)); // local lyrics: always
        assert!(!online_allowed_for_source(false, &g)); // online while unasked
        g.grant(Some(&d));
        assert!(online_allowed_for_source(false, &g));
        g.revoke(Some(&d));
        assert!(online_allowed_for_source(true, &g));
        assert!(!online_allowed_for_source(false, &g));
        let _ = std::fs::remove_dir_all(&d);
    }
}
