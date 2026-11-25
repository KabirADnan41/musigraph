//! Synchronised lyrics from LOCAL, user-supplied `.lrc` files. Pure parsing and lookup; no network, nothing
//! fabricated: if there is no file for the playing track (or it does not parse) there are no lyrics.
//!
//! Supported LRC: `[mm:ss.xx]text` (several time tags per line allowed), header tags (`[offset:+/-ms]`
//! honoured, `[ar:]`/`[ti:]`/... ignored) and the enhanced word tags `<mm:ss.xx>word`. Word timing is kept
//! only where the file really carries it (the renderer highlights words only then). Everything in the file
//! is UNTRUSTED text: size, line count, line length and timestamps are bounded and text is sanitised with
//! `sanitize_text` (Unicode, Bangla and RTL text survive; control and spoofing characters do not).

use crate::sanitize_text;
use std::path::{Path, PathBuf};

/// Largest `.lrc` file read (bytes).
pub const MAX_LRC_BYTES: usize = 512 * 1024;
/// Most lines kept from one file.
pub const MAX_LINES: usize = 4000;
/// Latest timestamp accepted (seconds): 24 hours.
const MAX_TIME_S: f64 = 86_400.0;
/// Most directory entries examined when looking for a file.
const MAX_DIR_ENTRIES: usize = 20_000;

#[derive(Clone, Debug, PartialEq)]
pub struct LyricWord {
    pub start_s: f64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LyricLine {
    pub start_s: f64,
    /// Sanitised text; empty for an instrumental gap.
    pub text: String,
    /// Word timings, only when the file provides them for this line; otherwise empty.
    pub words: Vec<LyricWord>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Lyrics {
    /// Sorted by `start_s`.
    pub lines: Vec<LyricLine>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrcError {
    TooLarge,
    NotUtf8,
    NoTimedLines,
}

/// Where playback is within the lyrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cursor {
    /// Line being sung (None before the first line starts).
    pub current: Option<usize>,
    pub next: Option<usize>,
    /// 0..1 progress through the current line (until the next line starts; 0 when unknown).
    pub progress: f32,
    /// Index of the word being sung, when the current line has word timing.
    pub word: Option<usize>,
}

/// `mm:ss`, `mm:ss.xx`, `mm:ss.xxx` or `mm:ss:xx` -> seconds (finite, 0..=24 h) or None.
fn parse_time(s: &str) -> Option<f64> {
    let (m, rest) = s.split_once(':')?;
    let m: u32 = m.trim().parse().ok()?;
    let rest = rest.trim().replace(':', ".");
    let sec: f64 = rest.parse().ok()?;
    let t = f64::from(m) * 60.0 + sec;
    (sec.is_finite() && (0.0..60.0 + 1e-9).contains(&sec) && (0.0..=MAX_TIME_S).contains(&t))
        .then_some(t)
}

/// Splits leading `[...]` tags off a line: returns (tags, remainder).
fn leading_tags(line: &str) -> (Vec<&str>, &str) {
    let mut tags = Vec::new();
    let mut rest = line.trim_start();
    while let Some(r) = rest.strip_prefix('[') {
        let Some(end) = r.find(']') else { break };
        tags.push(&r[..end]);
        rest = r[end + 1..].trim_start();
        if tags.len() >= 16 {
            break;
        }
    }
    (tags, rest)
}

/// Splits enhanced `<mm:ss.xx>word` markup into words; returns (plain text, words with timing).
fn split_words(text: &str, offset: f64) -> (String, Vec<LyricWord>) {
    if !text.contains('<') {
        return (sanitize_text(text), Vec::new());
    }
    let mut words = Vec::new();
    let mut plain = String::new();
    let mut cur_time: Option<f64> = None;
    let mut buf = String::new();
    let flush = |t: Option<f64>, b: &mut String, plain: &mut String, words: &mut Vec<LyricWord>| {
        let w = sanitize_text(b);
        if !w.is_empty() {
            if !plain.is_empty() {
                plain.push(' ');
            }
            plain.push_str(&w);
            if let Some(t) = t {
                words.push(LyricWord {
                    start_s: (t + offset).max(0.0),
                    text: w,
                });
            }
        }
        b.clear();
    };
    let mut rest = text;
    while let Some(i) = rest.find('<') {
        buf.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        match after
            .find('>')
            .and_then(|e| parse_time(&after[..e]).map(|t| (t, e)))
        {
            Some((t, e)) => {
                flush(cur_time, &mut buf, &mut plain, &mut words);
                cur_time = Some(t);
                rest = &after[e + 1..];
            }
            None => {
                buf.push('<');
                rest = after;
            }
        }
    }
    buf.push_str(rest);
    flush(cur_time, &mut buf, &mut plain, &mut words);
    // Word timing is only trusted when EVERY word has a timestamp and they do not run backwards.
    let plain_words = plain.split(' ').filter(|w| !w.is_empty()).count();
    let monotonic = words.windows(2).all(|w| w[1].start_s >= w[0].start_s);
    if words.len() != plain_words || !monotonic {
        words.clear();
    }
    (plain, words)
}

/// Parses a `.lrc` file's bytes.
pub fn parse(bytes: &[u8]) -> Result<Lyrics, LrcError> {
    if bytes.len() > MAX_LRC_BYTES {
        return Err(LrcError::TooLarge);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| LrcError::NotUtf8)?;
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    // Positive offset shifts lyrics EARLIER (the LRC convention); bounded to +-60 s. Read first so the tag may
    // appear anywhere in the file.
    let mut offset_s = 0.0f64;
    for raw in text.lines().take(MAX_LINES * 2) {
        for t in leading_tags(raw).0 {
            if let Some(ms) = t
                .strip_prefix("offset:")
                .and_then(|v| v.trim().parse::<f64>().ok())
            {
                if ms.is_finite() {
                    offset_s = (ms / 1000.0).clamp(-60.0, 60.0);
                }
            }
        }
    }
    let mut lines: Vec<LyricLine> = Vec::new();
    for raw in text.lines().take(MAX_LINES * 2) {
        let (tags, rest) = leading_tags(raw);
        if tags.is_empty() {
            continue;
        }
        let mut times = Vec::new();
        for t in &tags {
            if let Some(t) = parse_time(t) {
                times.push(t);
            }
        }
        if times.is_empty() {
            continue;
        }
        for t in times {
            let (plain, words) = split_words(rest, -offset_s);
            lines.push(LyricLine {
                start_s: (t - offset_s).max(0.0),
                text: plain,
                words,
            });
            if lines.len() >= MAX_LINES {
                break;
            }
        }
        if lines.len() >= MAX_LINES {
            break;
        }
    }
    if lines.is_empty() {
        return Err(LrcError::NoTimedLines);
    }
    lines.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    Ok(Lyrics { lines })
}

impl Lyrics {
    /// Cursor for a playback position in seconds. Stateless, so pausing (position frozen), resuming and
    /// seeking (position jumps) need no special handling.
    pub fn at(&self, pos_s: f64) -> Cursor {
        if !pos_s.is_finite() || self.lines.is_empty() {
            return Cursor {
                current: None,
                next: self.lines.first().map(|_| 0),
                progress: 0.0,
                word: None,
            };
        }
        // number of lines that have started
        let started = self.lines.partition_point(|l| l.start_s <= pos_s);
        let current = started.checked_sub(1);
        let next = (started < self.lines.len()).then_some(started);
        let (mut progress, mut word) = (0.0f32, None);
        if let Some(c) = current {
            let l = &self.lines[c];
            let end = next.map_or(l.start_s + 5.0, |n| self.lines[n].start_s);
            let span = (end - l.start_s).max(0.001);
            progress = (((pos_s - l.start_s) / span).clamp(0.0, 1.0)) as f32;
            if !l.words.is_empty() {
                let w = l.words.partition_point(|w| w.start_s <= pos_s);
                word = w.checked_sub(1);
            }
        }
        Cursor {
            current,
            next,
            progress,
            word,
        }
    }

    /// True if any line carries word-level timing.
    pub fn has_word_timing(&self) -> bool {
        self.lines.iter().any(|l| !l.words.is_empty())
    }
}

/// File-name form of a title/artist: characters Windows forbids in names and control characters are removed,
/// whitespace collapsed, lower-cased (Unicode-aware). Never contains a path separator.
pub fn file_key(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"/\\|?*".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect();
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Looks for `Artist - Title.lrc` or `Title.lrc` (case-insensitive, Unicode-aware) in `dir`. Only plain
/// files named `*.lrc` directly inside `dir` are considered (no recursion, no path joining of untrusted
/// text: the match is made on directory listings).
pub fn find(dir: &Path, title: &str, artist: &str) -> Option<PathBuf> {
    let t = file_key(title);
    if t.is_empty() {
        return None;
    }
    let a = file_key(artist);
    let wanted: Vec<String> = if a.is_empty() {
        vec![t]
    } else {
        vec![format!("{a} - {t}"), t]
    };
    let mut best: Option<(usize, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.take(MAX_DIR_ENTRIES).flatten() {
        let path = entry.path();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_none_or(|e| !e.eq_ignore_ascii_case("lrc"))
        {
            continue;
        }
        if !entry.file_type().is_ok_and(|ft| ft.is_file()) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let key = file_key(stem);
        if let Some(rank) = wanted.iter().position(|w| *w == key) {
            if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                best = Some((rank, path));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Reads and parses a lyrics file (bounded). `None` when it cannot be read or parsed.
pub fn load(path: &Path) -> Option<Lyrics> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_LRC_BYTES as u64 {
        return None;
    }
    parse(&std::fs::read(path).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\u{FEFF}[ar:Test Artist]\n[ti:Original Song]\n[offset:0]\n[00:05.00]First line\n[00:10.50]Second line\n[00:15.00][00:40.00]Chorus\n[00:20.00]\n[00:25.00]আমার সোনার বাংলা\n";

    #[test]
    fn parses_headers_multiple_tags_gaps_and_unicode() {
        let l = parse(SAMPLE.as_bytes()).unwrap();
        let starts: Vec<f64> = l.lines.iter().map(|x| x.start_s).collect();
        assert_eq!(starts, vec![5.0, 10.5, 15.0, 20.0, 25.0, 40.0]);
        assert_eq!(l.lines[2].text, "Chorus");
        assert_eq!(l.lines[5].text, "Chorus");
        assert_eq!(l.lines[3].text, "", "instrumental gap kept");
        assert_eq!(l.lines[4].text, "আমার সোনার বাংলা");
    }

    #[test]
    fn cursor_follows_position_pause_and_seek_without_state() {
        let l = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(l.at(0.0).current, None);
        assert_eq!(l.at(0.0).next, Some(0));
        let c = l.at(12.0);
        assert_eq!((c.current, c.next), (Some(1), Some(2)));
        assert!((c.progress - (12.0 - 10.5) / 4.5).abs() < 1e-3);
        // paused: the same position gives the same answer; seek back gives the earlier line
        assert_eq!(l.at(12.0), l.at(12.0));
        assert_eq!(l.at(6.0).current, Some(0));
        assert_eq!(l.at(1000.0).current, Some(5));
        assert_eq!(l.at(1000.0).next, None);
        assert_eq!(l.at(f64::NAN).current, None);
    }

    #[test]
    fn offset_shifts_lyrics_earlier_when_positive_and_is_bounded() {
        let l = parse(b"[offset:+500]\n[00:10.00]x\n").unwrap();
        assert!((l.lines[0].start_s - 9.5).abs() < 1e-9);
        let l = parse(b"[offset:-250]\n[00:10.00]x\n").unwrap();
        assert!((l.lines[0].start_s - 10.25).abs() < 1e-9);
        let l = parse(b"[offset:999999999]\n[00:10.00]x\n").unwrap();
        assert_eq!(l.lines[0].start_s, 0.0, "clamped to 60 s and floored at 0");
    }

    #[test]
    fn word_timing_is_kept_only_when_every_word_has_a_timestamp() {
        let l = parse(b"[00:10.00]<00:10.00>Hello <00:10.50>big <00:11.00>world\n[00:12.00]<00:12.00>Only <oops>half timed\n[00:14.00]plain words\n").unwrap();
        assert_eq!(l.lines[0].text, "Hello big world");
        assert_eq!(l.lines[0].words.len(), 3);
        assert!(
            l.lines[1].words.is_empty(),
            "partly timed line gets line-level only"
        );
        assert!(l.lines[2].words.is_empty());
        assert!(l.has_word_timing());
        let c = l.at(10.7);
        assert_eq!((c.current, c.word), (Some(0), Some(1)));
    }

    #[test]
    fn hostile_input_is_bounded_and_never_panics() {
        assert_eq!(
            parse(&vec![b'a'; MAX_LRC_BYTES + 1]),
            Err(LrcError::TooLarge)
        );
        assert_eq!(parse(&[0xFF, 0xFE, 0x00]), Err(LrcError::NotUtf8));
        assert_eq!(
            parse(b"no timestamps here\njust text"),
            Err(LrcError::NoTimedLines)
        );
        assert_eq!(parse(b""), Err(LrcError::NoTimedLines));
        let many = "[00:01.00]x\n".repeat(MAX_LINES * 3);
        assert_eq!(parse(many.as_bytes()).unwrap().lines.len(), MAX_LINES);
        // absurd or malformed timestamps are ignored, control/bidi characters stripped from text
        let l = parse(
            "[99999:99.99]late\n[00:61.00]bad\n[ab:cd]x\n[00:02.00]a\u{202E}b\u{0}c\n[-1:00]neg\n"
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(l.lines.len(), 1);
        assert_eq!(l.lines[0].text, "a b c");
        let tags = "[00:01.00]".repeat(100) + "x";
        assert!(parse(tags.as_bytes()).unwrap().lines.len() <= 16);
    }

    #[test]
    fn file_keys_are_safe_and_lookup_is_case_insensitive_unicode_and_non_recursive() {
        assert_eq!(file_key("AC/DC: Back\\In  Black?"), "ac dc back in black");
        assert_eq!(file_key("../../etc/passwd"), ".. .. etc passwd");
        let dir = std::env::temp_dir().join(format!("mg-lrc-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(
            dir.join("Aster Vale - Midnight Harbour.LRC"),
            "[00:01.00]hi\n",
        )
        .unwrap();
        std::fs::write(dir.join("নীল আকাশের গান.lrc"), "[00:01.00]hi\n").unwrap();
        std::fs::write(dir.join("sub").join("Deep.lrc"), "[00:01.00]hi\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        let f = find(&dir, "MIDNIGHT harbour", "aster vale").unwrap();
        assert!(f.ends_with("Aster Vale - Midnight Harbour.LRC"));
        assert!(
            find(&dir, "Midnight Harbour", "").is_none(),
            "title-only needs a title-only file"
        );
        assert!(
            find(&dir, "নীল আকাশের গান", "সুরেলা").is_some(),
            "falls back to the title-only name"
        );
        assert!(find(&dir, "Deep", "").is_none(), "no recursion");
        assert!(find(&dir, "", "x").is_none());
        assert!(find(&dir.join("missing"), "x", "y").is_none());
        assert!(load(&f).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
