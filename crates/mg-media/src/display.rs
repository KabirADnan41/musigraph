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

const VERSION_WORDS: [&str; 9] = [
    "remaster",
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

fn is_version(s: &str) -> bool {
    let l = s.to_lowercase();
    VERSION_WORDS.iter().any(|w| l.contains(w))
}

/// Splits a feat. credit off `s`: returns (rest, featured artists) if found as a bracketed or trailing credit.
fn split_feat(s: &str) -> Option<(String, String)> {
    let l = s.to_lowercase();
    for (open, close) in [('(', ')'), ('[', ']')] {
        if let Some(i) = l.find(open)
            && let Some(j) = l[i..].find(close)
        {
            let inner = &s[i + 1..i + j];
            let il = inner.to_lowercase();
            if let Some(f) = FEAT.iter().find(|f| il.starts_with(**f)) {
                let rest = format!("{}{}", s[..i].trim_end(), &s[i + j + 1..]);
                return Some((rest.trim().to_string(), inner[f.len()..].trim().to_string()));
            }
        }
    }
    for f in FEAT {
        if let Some(i) = l.find(&format!(" {f}")) {
            return Some((
                s[..i].trim().to_string(),
                s[i + 1 + f.len()..].trim().to_string(),
            ));
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
            stripped
                .rfind(['(', '['])
                .filter(|&i| is_version(&stripped[i + 1..]))
                .map(|i| (i, stripped[i + 1..].trim().to_string()))
        } else {
            t.rfind(" - ")
                .filter(|&i| is_version(&t[i + 3..]))
                .map(|i| (i, t[i + 3..].trim().to_string()))
        };
        match cut {
            Some((i, v)) if !t[..i].trim().is_empty() => {
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
}
