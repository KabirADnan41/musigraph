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
    pub fn decline() -> Consent {
        Self::decline_to(app_dir().as_deref())
    }

    pub fn grant_to(dir: Option<&Path>) -> Consent {
        // A failed save still returns the in-memory choice for this session.
        if let Some(d) = dir {
            let _ = save_to(d, Consent::Granted);
        }
        Consent::Granted
    }

    pub fn decline_to(dir: Option<&Path>) -> Consent {
        if let Some(d) = dir {
            let _ = save_to(d, Consent::Declined);
        }
        Consent::Declined
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
        assert_eq!(Consent::decline_to(Some(&d)), Consent::Declined);
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
        assert_eq!(PROMPT, "Enable automatic online lyrics?\nMusigraph can retrieve lyrics from LRCLIB using the currently playing track's title, artist, album and duration. These details, your connection IP address and the Musigraph application identifier will be visible to the service.\nNo account is required. Musigraph does not upload your audio or maintain a listening history. LRCLIB's data-retention practices have not been verified.\nYou can disable online lyrics at any time.");
    }
}
