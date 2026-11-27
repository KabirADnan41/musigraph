//! Optional online lyrics from LRCLIB (`GET https://lrclib.net/api/get`), owner-approved 2026-11-27 for
//! local development ONLY behind an explicit opt-in (consent UI and persistence live elsewhere).
//!
//! Purpose: turn the playing track into at most one strict lookup and turn the reply into lyrics or a
//! clear "no reliable lyrics" result. Pure and clock-injected except `client` (worker thread) and `winrt`
//! (the Windows HTTP transport).
//!
//! Trust boundary: everything from the network is UNTRUSTED. The body is size-capped, JSON depth-checked,
//! fields type-checked, and lyrics go through the bounded `lrc` parser / `sanitize_text`.
//!
//! Privacy invariants: only title, artist, album, duration and a User-Agent are sent, to one HTTPS host;
//! nothing is written to disk; nothing here logs; `Debug` output of queries and lyrics is redacted.
//! Without consent no request is ever produced; revoking consent cancels, invalidates and clears.

mod client;
#[cfg(windows)]
mod winrt;

pub use client::LrclibClient;
#[cfg(windows)]
pub use winrt::WinRtTransport;

use crate::lrc::{self, Lyrics};
use crate::{NowPlaying, sanitize_text};
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

/// The only endpoint used.
pub const ENDPOINT: &str = "https://lrclib.net/api/get";
/// Every URL a transport may fetch starts with this (HTTPS, host pinned).
pub const ALLOWED_PREFIX: &str = "https://lrclib.net/api/";
/// Largest response body accepted (bytes).
pub const MAX_BODY_BYTES: usize = 512 * 1024;
/// Returned duration must be within this many seconds of the playing track (LRCLIB's own rule).
pub const DURATION_TOLERANCE_S: f64 = 2.0;
/// LRCLIB accepts durations 1..=3600 s.
pub const MAX_DURATION_S: u32 = 3600;
/// Quiet time after a track change before a lookup (skipping through tracks sends nothing).
pub const DEBOUNCE: Duration = Duration::from_millis(1500);
/// Global cap, far below LRCLIB's documented ~45/min for generic clients.
pub const MAX_REQUESTS_PER_MINUTE: usize = 10;
/// App-side deadline for one whole request (headers + body).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(8);
const RATE_WINDOW: Duration = Duration::from_secs(60);
const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(60);
const MIN_RETRY_AFTER: Duration = Duration::from_secs(1);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(600);
/// Small fixed margin on top of Retry-After (stands in for the jitter LRCLIB suggests).
const RETRY_MARGIN: Duration = Duration::from_secs(1);
const BACKOFF_MIN: Duration = Duration::from_secs(10);
const BACKOFF_MAX: Duration = Duration::from_secs(300);
/// Settled track keys remembered per session (RAM only); cleared wholesale when full.
const MAX_SETTLED: usize = 1024;
/// Deepest JSON nesting accepted before handing the body to the system parser.
const MAX_JSON_DEPTH: usize = 16;

/// `User-Agent` sent with every request (LRCLIB requires client identification).
pub fn user_agent() -> String {
    format!(
        "Musigraph/{} (+https://github.com/KabirADnan41/musigraph)",
        env!("CARGO_PKG_VERSION")
    )
}

/// True only for HTTPS URLs on the approved host and API path.
pub fn is_allowed_url(url: &str) -> bool {
    url.starts_with(ALLOWED_PREFIX)
}

/// One strict lookup key: never title-only, duration always present. `Debug` is redacted.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Query {
    title: String,
    artist: String,
    album: String,
    duration_s: u32,
}

impl fmt::Debug for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Query {{ duration_s: {} }}", self.duration_s)
    }
}

impl Query {
    /// `None` when title or artist is empty after sanitising, or the duration is missing, non-finite
    /// or outside 1..=3600 s once rounded.
    pub fn new(title: &str, artist: &str, album: &str, duration_s: Option<f64>) -> Option<Self> {
        let (title, artist) = (sanitize_text(title), sanitize_text(artist));
        let d = duration_s?.round(); // NaN stays NaN and fails the range check
        if title.is_empty() || artist.is_empty() || !(1.0..=f64::from(MAX_DURATION_S)).contains(&d)
        {
            return None;
        }
        Some(Self {
            title,
            artist,
            album: sanitize_text(album),
            duration_s: d as u32,
        })
    }

    pub fn from_now_playing(np: &NowPlaying) -> Option<Self> {
        Self::new(&np.title, &np.artist, &np.album, np.duration_s)
    }

    pub fn duration_s(&self) -> u32 {
        self.duration_s
    }

    /// The request URL (percent-encoded query; album omitted when unknown).
    pub fn url(&self) -> String {
        let mut u = format!("{ENDPOINT}?track_name=");
        encode(&self.title, &mut u);
        u.push_str("&artist_name=");
        encode(&self.artist, &mut u);
        if !self.album.is_empty() {
            u.push_str("&album_name=");
            encode(&self.album, &mut u);
        }
        u.push_str(&format!("&duration={}", self.duration_s));
        u
    }
}

/// RFC 3986 percent-encoding of UTF-8 bytes; only unreserved characters pass through.
fn encode(s: &str, out: &mut String) {
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
}

/// Fields read from an LRCLIB record (already type-checked; still untrusted).
#[derive(Clone, Default, PartialEq)]
pub struct Record {
    pub track_name: Option<String>,
    pub album_name: Option<String>,
    pub duration: Option<f64>,
    pub instrumental: Option<bool>,
    pub synced: Option<String>,
    pub plain: Option<String>,
}

/// Result of one lookup. `Debug` is redacted (line counts only).
#[derive(Clone, PartialEq)]
pub enum Outcome {
    /// Time-synchronised lyrics.
    Synced(Lyrics),
    /// Plain lyrics only: for the labelled "Not synchronized" static panel. Sanitised lines.
    Unsynced(Vec<String>),
    /// 404: no record.
    NotFound,
    /// A record came back but is not a reliable match (duration, instrumental, version words, no lyrics).
    Rejected,
    /// 429/503: wait this long (bounded) before any further request.
    RetryAfter(Duration),
    /// Policy refusal: redirect, disallowed URL, oversized body.
    Refused,
    /// Network failure, timeout, malformed body or unexpected status.
    Error,
}

impl fmt::Debug for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Synced(l) => write!(f, "Synced({} lines)", l.lines.len()),
            Self::Unsynced(l) => write!(f, "Unsynced({} lines)", l.len()),
            Self::NotFound => f.write_str("NotFound"),
            Self::Rejected => f.write_str("Rejected"),
            Self::RetryAfter(d) => write!(f, "RetryAfter({d:?})"),
            Self::Refused => f.write_str("Refused"),
            Self::Error => f.write_str("Error"),
        }
    }
}

/// What a transport hands back. The body is only read for status 200. `Debug` is redacted.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub retry_after: Option<Duration>,
    pub body: Vec<u8>,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HttpResponse {{ status: {}, body: {} bytes }}",
            self.status,
            self.body.len()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportError {
    Cancelled,
    Timeout,
    /// Body (or declared Content-Length) above `MAX_BODY_BYTES`.
    TooLarge,
    /// URL is not HTTPS on the approved host.
    NotAllowed,
    Network,
}

/// One blocking HTTPS GET. Implementations must: refuse `!is_allowed_url`, never follow redirects,
/// stop reading past `MAX_BODY_BYTES`, give up after `REQUEST_TIMEOUT`, and abort promptly once
/// `cancelled()` returns true.
pub trait Transport {
    fn get(&self, url: &str, cancelled: &dyn Fn() -> bool) -> Result<HttpResponse, TransportError>;
}

/// Maps a transport result to an outcome (status policy, size cap, JSON, strict match).
pub fn evaluate(q: &Query, r: Result<HttpResponse, TransportError>) -> Outcome {
    let resp = match r {
        Ok(resp) => resp,
        Err(TransportError::TooLarge | TransportError::NotAllowed) => return Outcome::Refused,
        Err(_) => return Outcome::Error,
    };
    match resp.status {
        200 if resp.body.len() > MAX_BODY_BYTES => Outcome::Refused,
        200 => parse_record(&resp.body).map_or(Outcome::Error, |rec| judge(q, &rec)),
        404 => Outcome::NotFound,
        429 | 503 => Outcome::RetryAfter(
            resp.retry_after
                .unwrap_or(DEFAULT_RETRY_AFTER)
                .clamp(MIN_RETRY_AFTER, MAX_RETRY_AFTER),
        ),
        300..=399 => Outcome::Refused, // redirects are never followed
        _ => Outcome::Error,
    }
}

/// Strict match of a parsed record against the query.
pub fn judge(q: &Query, rec: &Record) -> Outcome {
    let duration_ok = rec
        .duration
        .is_some_and(|d| (d - f64::from(q.duration_s)).abs() <= DURATION_TOLERANCE_S); // NaN fails
    let Some(track) = &rec.track_name else {
        return Outcome::Rejected; // ambiguous
    };
    let theirs = format!("{track} {}", rec.album_name.as_deref().unwrap_or(""));
    if !duration_ok
        || rec.instrumental == Some(true)
        || version_words(&format!("{} {}", q.title, q.album)) != version_words(&theirs)
    {
        return Outcome::Rejected;
    }
    if let Some(Ok(l)) = rec.synced.as_ref().map(|s| lrc::parse(s.as_bytes())) {
        return Outcome::Synced(l);
    }
    let lines: Vec<String> = rec
        .plain
        .as_deref()
        .unwrap_or("")
        .lines()
        .take(lrc::MAX_LINES)
        .map(sanitize_text)
        .collect();
    if lines.iter().all(String::is_empty) {
        return Outcome::Rejected;
    }
    Outcome::Unsynced(lines)
}

/// Recording-distinguishing words; a difference between the playing track and the record means a
/// different recording (live vs studio, remix, ...) and the record is rejected. Whole words only.
fn version_words(s: &str) -> Vec<&'static str> {
    const WORDS: [(&str, &str); 14] = [
        ("live", "live"),
        ("remix", "remix"),
        ("remixed", "remix"),
        ("mix", "mix"),
        ("acoustic", "acoustic"),
        ("cover", "cover"),
        ("instrumental", "instrumental"),
        ("karaoke", "karaoke"),
        ("remaster", "remaster"),
        ("remastered", "remaster"),
        ("extended", "extended"),
        ("edit", "edit"),
        ("unplugged", "unplugged"),
        ("demo", "demo"),
    ];
    let mut found: Vec<&'static str> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter_map(|w| {
            WORDS
                .iter()
                .find(|(k, _)| w.eq_ignore_ascii_case(k))
                .map(|(_, v)| *v)
        })
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// Rejects JSON nested deeper than `MAX_JSON_DEPTH` (brackets outside strings).
fn json_depth_ok(text: &str) -> bool {
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for b in text.bytes() {
        match (in_str, esc, b) {
            (true, true, _) => esc = false,
            (true, false, b'\\') => esc = true,
            (true, false, b'"') | (false, _, b'"') => in_str = !in_str,
            (false, _, b'[' | b'{') => {
                depth += 1;
                if depth > MAX_JSON_DEPTH {
                    return false;
                }
            }
            (false, _, b']' | b'}') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    true
}

/// Parses an LRCLIB `/api/get` body with the system JSON parser. `None` for anything malformed.
/// Fields of the wrong type are treated as absent.
#[cfg(windows)]
pub fn parse_record(body: &[u8]) -> Option<Record> {
    use windows::Data::Json::{JsonObject, JsonValueType};
    use windows::core::HSTRING;
    let text = std::str::from_utf8(body).ok()?;
    if body.len() > MAX_BODY_BYTES || !json_depth_ok(text) {
        return None;
    }
    let mut obj = None;
    if !JsonObject::TryParse(&HSTRING::from(text), &mut obj).ok()? {
        return None;
    }
    let obj = obj?;
    let field = |k: &str, t: JsonValueType| {
        obj.Lookup(&HSTRING::from(k))
            .ok()
            .filter(|v| v.ValueType().ok() == Some(t))
    };
    let string = |k| {
        field(k, JsonValueType::String)
            .and_then(|v| v.GetString().ok())
            .map(|h| String::from_utf16_lossy(&h))
    };
    Some(Record {
        track_name: string("trackName"),
        album_name: string("albumName"),
        duration: field("duration", JsonValueType::Number).and_then(|v| v.GetNumber().ok()),
        instrumental: field("instrumental", JsonValueType::Boolean)
            .and_then(|v| v.GetBoolean().ok()),
        synced: string("syncedLyrics"),
        plain: string("plainLyrics"),
    })
}

/// shortcut: JSON parsing uses Windows.Data.Json only; add a parser if a non-Windows build needs this.
#[cfg(not(windows))]
pub fn parse_record(_body: &[u8]) -> Option<Record> {
    None
}

/// A request the requester wants sent, tagged with the consent generation it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub query: Query,
    pub generation: u64,
}

/// Decides WHEN to ask. Pure; the caller injects `now`. At most one request in flight.
#[derive(Debug)]
pub struct Requester {
    consent: bool,
    generation: u64,
    current: Option<(Query, Instant)>,
    in_flight: Option<Request>,
    /// Tracks already answered this session (positive for the current track, or any negative).
    settled: HashSet<Query>,
    sent: VecDeque<Instant>,
    paused_until: Option<Instant>,
    backoff: Duration,
}

impl Default for Requester {
    fn default() -> Self {
        Self {
            consent: false,
            generation: 0,
            current: None,
            in_flight: None,
            settled: HashSet::new(),
            sent: VecDeque::new(),
            paused_until: None,
            backoff: BACKOFF_MIN,
        }
    }
}

impl Requester {
    pub fn consented(&self) -> bool {
        self.consent
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn in_flight(&self) -> Option<&Request> {
        self.in_flight.as_ref()
    }

    /// Turning consent off is REVOKE: new generation (in-flight work and late replies become stale)
    /// and all track memory is dropped. Rate/pause state is kept so toggling cannot bypass limits.
    pub fn set_consent(&mut self, on: bool) {
        if !on {
            self.generation = self.generation.wrapping_add(1);
            self.current = None;
            self.in_flight = None;
            self.settled = HashSet::new(); // release memory, not just length
        }
        self.consent = on;
    }

    /// Reports the playing track (or none). A different key restarts the debounce.
    pub fn observe(&mut self, now: Instant, track: Option<&Query>) {
        if self.current.as_ref().map(|(q, _)| q) != track {
            self.current = track.map(|q| (q.clone(), now));
        }
    }

    /// The request to send now, if any. Never one without consent.
    pub fn poll(&mut self, now: Instant) -> Option<Request> {
        if !self.consent || self.in_flight.is_some() || self.paused_until.is_some_and(|t| now < t) {
            return None;
        }
        let (q, since) = self.current.as_ref()?;
        if now.saturating_duration_since(*since) < DEBOUNCE || self.settled.contains(q) {
            return None;
        }
        while self
            .sent
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= RATE_WINDOW)
        {
            self.sent.pop_front();
        }
        if self.sent.len() >= MAX_REQUESTS_PER_MINUTE {
            return None;
        }
        self.sent.push_back(now);
        let req = Request {
            query: q.clone(),
            generation: self.generation,
        };
        self.in_flight = Some(req.clone());
        Some(req)
    }

    /// The request could not be queued; it will be retried by a later `poll` (rate cap still applies).
    pub fn on_submit_failed(&mut self, req: &Request) {
        if self.in_flight.as_ref() == Some(req) {
            self.in_flight = None;
        }
    }

    /// Feeds a reply back. Returns the outcome only if it is for the current generation AND the current
    /// track; stale replies are dropped.
    pub fn on_outcome(&mut self, now: Instant, req: &Request, outcome: Outcome) -> Option<Outcome> {
        if req.generation != self.generation || self.in_flight.as_ref() != Some(req) {
            return None;
        }
        self.in_flight = None;
        match outcome {
            Outcome::RetryAfter(d) => {
                // not settled: the same track is asked again once the pause is over
                self.paused_until = Some(now + d + RETRY_MARGIN);
                return None;
            }
            Outcome::Error => {
                self.paused_until = Some(now + self.backoff);
                self.backoff = (self.backoff * 2).min(BACKOFF_MAX);
            }
            _ => self.backoff = BACKOFF_MIN,
        }
        let is_current = self.current.as_ref().map(|(q, _)| q) == Some(&req.query);
        if !is_current && matches!(outcome, Outcome::Synced(_) | Outcome::Unsynced(_)) {
            return None; // lyrics for a track no longer playing are not kept
        }
        if self.settled.len() >= MAX_SETTLED {
            self.settled.clear();
        }
        self.settled.insert(req.query.clone());
        is_current.then_some(outcome)
    }
}

#[cfg(test)]
mod tests;
