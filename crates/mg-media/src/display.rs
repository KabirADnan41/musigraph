//! Display-only rewriting of track metadata for typography (Album Focus spec, 2026-11-27).
//!
//! Version suffixes such as "(Remastered 2011)" or "- Radio Edit" move from the title to a quieter meta line.
//! A "feat." credit moves to the artist line. More than three artists shorten to "A, B & 2 more".
//! The raw strings, which lyric matching and every other lookup use, are never changed: this only decides
//! what is drawn. The result never loses text the user would need: when stripping would empty the title,
//! the original title is kept.

/// What to draw for one track.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DisplayMeta {
    pub title: String,
    pub artist: String,
    /// Version information moved out of the title ("Remastered 2011", "Live", "Radio Edit"); empty if none.
    pub version: String,
}

const VERSION_WORDS: [&str; 10] = [
    "remaster",
    "remastered",
    "radio edit",
    "live",
    "deluxe",
    "single version",
    "album version",
    "mono",
    "stereo",
    "anniversary",
];
const FEAT: [&str; 4] = ["feat. ", "feat ", "ft. ", "featuring "];

/// True when `s` contains a version word as a whole word (split on non-alphanumerics, ASCII-case-insensitive);
/// multi-word entries match as consecutive word sequences. "Alive" and "Delivery" do not contain "live".
fn is_version(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    let words: Vec<&str> = l
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    VERSION_WORDS.iter().any(|entry| {
        let seq: Vec<&str> = entry.split(' ').collect();
        words.windows(seq.len()).any(|w| w == seq.as_slice())
    })
}

/// Splits a feat. credit off `s`: returns (rest, featured artists) if found as a bracketed or trailing credit.
/// Matching runs on `to_ascii_lowercase`, which keeps byte offsets identical to `s` (Unicode lowercasing does
/// not: U+0130 grows, so offsets from it would slice `s` mid-character). Every slice uses `get`.
fn split_feat(s: &str) -> Option<(String, String)> {
    let l = s.to_ascii_lowercase();
    for (open, close) in [('(', ')'), ('[', ']')] {
        if let Some(i) = l.find(open)
            && let Some(j) = l.get(i..).and_then(|t| t.find(close))
            && let Some(inner) = s.get(i + 1..i + j)
            && let Some(il) = l.get(i + 1..i + j)
            && let Some(f) = FEAT.iter().find(|f| il.starts_with(**f))
            && let Some(credit) = inner.get(f.len()..)
        {
            let rest = format!("{}{}", s.get(..i)?.trim_end(), s.get(i + j + 1..)?);
            return Some((rest.trim().to_string(), credit.trim().to_string()));
        }
    }
    for f in FEAT {
        if let Some(i) = l.find(&format!(" {f}"))
            && let Some(head) = s.get(..i)
            && let Some(credit) = s.get(i + 1 + f.len()..)
        {
            return Some((head.trim().to_string(), credit.trim().to_string()));
        }
    }
    None
}

/// Moves trailing version suffixes ("(…Remaster…)", "[Live]", "- Radio Edit") out of the title.
fn split_version(mut title: String) -> (String, Vec<String>) {
    let mut found = Vec::new();
    loop {
        let t = title.trim_end();
        let cut = if let Some(stripped) = t.strip_suffix(')').or_else(|| t.strip_suffix(']')) {
            stripped.rfind(['(', '[']).and_then(|i| {
                let v = stripped.get(i + 1..)?;
                is_version(v).then(|| (i, v.trim().to_string()))
            })
        } else {
            t.rfind(" - ").and_then(|i| {
                let v = t.get(i + 3..)?;
                is_version(v).then(|| (i, v.trim().to_string()))
            })
        };
        match cut {
            Some((i, v)) if t.get(..i).is_some_and(|h| !h.trim().is_empty()) => {
                found.insert(0, v);
                title = t[..i].trim_end().to_string();
            }
            _ => return (t.to_string(), found),
        }
    }
}

fn shorten_artists(artist: &str) -> String {
    let parts: Vec<&str> = artist
        .split([',', ';', '&'])
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if parts.len() > 3 {
        format!("{}, {} & {} more", parts[0], parts[1], parts.len() - 2)
    } else {
        artist.trim().to_string()
    }
}

/// Builds the display strings. Inputs are already sanitised by the media layer.
pub fn display_meta(title: &str, artist: &str) -> DisplayMeta {
    let (mut t, mut feat) = (title.trim().to_string(), None);
    if let Some((rest, f)) = split_feat(&t)
        && !rest.is_empty()
        && !f.is_empty()
    {
        t = rest;
        feat = Some(f);
    }
    let (t, versions) = split_version(t);
    let mut a = shorten_artists(artist);
    if let Some(f) = feat
        && !a.to_lowercase().contains(&f.to_lowercase())
    {
        a = if a.is_empty() {
            f
        } else {
            format!("{a} feat. {f}")
        };
    }
    DisplayMeta {
        title: if t.is_empty() {
            title.trim().to_string()
        } else {
            t
        },
        artist: a,
        version: versions.join(" · "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(t: &str, a: &str) -> (String, String, String) {
        let m = display_meta(t, a);
        (m.title, m.artist, m.version)
    }

    #[test]
    fn version_suffixes_move_to_the_meta_line() {
        assert_eq!(d("Heroes (Remastered 2017)", "Bowie").0, "Heroes");
        assert_eq!(d("Heroes (Remastered 2017)", "Bowie").2, "Remastered 2017");
        assert_eq!(d("Song - 2011 Remaster", "X").0, "Song");
        assert_eq!(
            d("Song - Radio Edit", "X"),
            ("Song".into(), "X".into(), "Radio Edit".into())
        );
        assert_eq!(
            d("Song [Live] (Deluxe Edition)", "X").2,
            "Live · Deluxe Edition"
        );
        // ordinary brackets and dashes stay
        assert_eq!(d("Song (Interlude)", "X").0, "Song (Interlude)");
        assert_eq!(d("Part 1 - The Beginning", "X").0, "Part 1 - The Beginning");
        // never empties the title
        assert_eq!(d("(Live)", "X").0, "(Live)");
    }

    #[test]
    fn featured_artists_move_to_the_artist_line() {
        assert_eq!(
            d("Song (feat. B)", "A"),
            ("Song".into(), "A feat. B".into(), String::new())
        );
        assert_eq!(d("Song ft. B", "A").1, "A feat. B");
        assert_eq!(
            d("Song [Featuring B] (Remastered)", "A"),
            ("Song".into(), "A feat. B".into(), "Remastered".into())
        );
        // already credited: not duplicated
        assert_eq!(d("Song (feat. B)", "A, B").1, "A, B");
        // a word merely starting with "ft" is not a credit
        assert_eq!(d("Left Behind", "A").0, "Left Behind");
    }

    #[test]
    fn long_artist_lists_shorten_and_unicode_is_untouched() {
        assert_eq!(d("S", "A, B, C, D, E").1, "A, B & 3 more");
        assert_eq!(d("S", "A, B & C").1, "A, B & C");
        assert_eq!(d("আমার সোনার বাংলা", "রবীন্দ্রনাথ ঠাকুর").0, "আমার সোনার বাংলা");
        assert_eq!(
            d("مرحبا (Live)", "فنان"),
            ("مرحبا".into(), "فنان".into(), "Live".into())
        );
        assert_eq!(d("", ""), (String::new(), String::new(), String::new()));
    }

    #[test]
    fn unicode_lowercase_length_changes_cannot_panic_or_drop_credits() {
        // U+0130 lowercases to two chars (2 bytes to 3), U+212A (Kelvin) to ASCII 'k' (3 bytes to 1)
        assert_eq!(
            d("İİİ Song [feat. X]", "A"),
            ("İİİ Song".into(), "A feat. X".into(), String::new())
        );
        assert_eq!(d("İstanbul (feat. B)", "A").0, "İstanbul");
        assert_eq!(d("İstanbul (feat. B)", "A").1, "A feat. B");
        assert_eq!(d("\u{212A} Song ft. B", "A").0, "\u{212A} Song");
        assert_eq!(d("Song (feat. İ)", "A").1, "A feat. İ");
        assert_eq!(d("e\u{301}e\u{301} (Live)", "A").2, "Live");
        assert_eq!(d("🎵 Song (feat. 🎤)", "A").1, "A feat. 🎤");
    }

    #[test]
    fn version_words_match_whole_words_only() {
        for t in [
            "Alive",
            "Olive",
            "Delivery",
            "Monopoly",
            "Stereotype",
            "Remasterpiece",
        ] {
            let paren = format!("Song ({t})");
            let dash = format!("Song - {t}");
            assert_eq!(d(&paren, "X").0, paren, "{t}");
            assert_eq!(d(&dash, "X").0, dash, "{t}");
        }
        assert_eq!(d("Song (Remastered)", "X").2, "Remastered");
        assert_eq!(d("Song (LIVE at Wembley)", "X").2, "LIVE at Wembley");
        assert_eq!(d("Song - Single Version", "X").2, "Single Version");
        assert_eq!(d("Song [Album Version]", "X").2, "Album Version");
        assert_eq!(
            d("Song (25th Anniversary Mix)", "X").2,
            "25th Anniversary Mix"
        );
        assert_eq!(d("Song (Mono)", "X").2, "Mono");
        assert!(!is_version("single other version"));
    }

    #[test]
    fn mixed_unicode_never_panics() {
        let parts = [
            "İ", "\u{212A}", "e\u{301}", "🎵", "(", ")", "[", "]", " - ", " ft. ", "feat. ",
            "(feat. ", "[Live]", "ß", "আ", "م", "ǅ", " ", "x",
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for _ in 0..3000 {
            let mut s = String::new();
            for _ in 0..next() % 9 + 1 {
                s.push_str(parts[next() % parts.len()]);
            }
            let _ = display_meta(&s, &s);
        }
        let _ = display_meta(&"İ(feat. ".repeat(20_000), "A");
        let _ = display_meta(&"İ".repeat(100_000), "A, B, C, D");
    }
}
