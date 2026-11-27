//! Core tests. Synthetic text only; no network.

use super::*;

const SYNCED: &str = "[00:01.00]la la la\n[00:03.50]na na";
const PLAIN: &str = "la la la\nna na";

fn q() -> Query {
    Query::new("Synthetic Song", "Test Artist", "Test Album", Some(200.0)).unwrap()
}

fn qn(i: u32) -> Query {
    Query::new(&format!("Synthetic {i}"), "Test Artist", "", Some(200.0)).unwrap()
}

fn rec(duration: f64) -> Record {
    Record {
        track_name: Some("Synthetic Song".into()),
        album_name: Some("Test Album".into()),
        duration: Some(duration),
        instrumental: Some(false),
        synced: Some(SYNCED.into()),
        plain: Some(PLAIN.into()),
    }
}

fn ok(status: u16, body: &str) -> Result<HttpResponse, TransportError> {
    Ok(HttpResponse {
        status,
        retry_after: None,
        body: body.as_bytes().to_vec(),
    })
}

fn json(duration: &str) -> String {
    format!(
        r#"{{"id":1,"trackName":"Synthetic Song","artistName":"Test Artist","albumName":"Test Album","duration":{duration},"instrumental":false,"plainLyrics":"la la la\nna na","syncedLyrics":"[00:01.00]la la la\n[00:03.50]na na"}}"#
    )
}

// ---- request building ----

#[test]
fn query_refuses_missing_title_artist_or_bad_duration() {
    let d = Some(200.0);
    assert!(Query::new("", "A", "", d).is_none());
    assert!(Query::new("  \n\t", "A", "", d).is_none());
    assert!(Query::new("T", "", "", d).is_none(), "never title-only");
    assert!(Query::new("T", "\u{202e}", "", d).is_none());
    for bad in [
        None,
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(0.0),
        Some(0.49),
        Some(-5.0),
        Some(3600.5),
        Some(1e300),
        Some(f64::MAX),
    ] {
        assert!(Query::new("T", "A", "", bad).is_none(), "{bad:?}");
    }
    assert_eq!(Query::new("T", "A", "", Some(0.5)).unwrap().duration_s(), 1);
    assert_eq!(
        Query::new("T", "A", "", Some(3600.4)).unwrap().duration_s(),
        3600
    );
    assert_eq!(
        Query::new("T", "A", "", Some(199.6)).unwrap().duration_s(),
        200
    );
}

#[test]
fn url_is_percent_encoded_and_pinned() {
    let q = Query::new("A&B=C ?#/+%", "Ünïcödé 歌", "", Some(199.6)).unwrap();
    assert_eq!(
        q.url(),
        "https://lrclib.net/api/get?track_name=A%26B%3DC%20%3F%23%2F%2B%25\
         &artist_name=%C3%9Cn%C3%AFc%C3%B6d%C3%A9%20%E6%AD%8C&duration=200"
    );
    let q = Query::new("a-b.c_d~e", "x\ny", "Alb um", Some(61.0)).unwrap();
    assert_eq!(
        q.url(),
        "https://lrclib.net/api/get?track_name=a-b.c_d~e&artist_name=x%20y&album_name=Alb%20um&duration=61"
    );
    assert!(is_allowed_url(&q.url()));
    assert!(q.url().starts_with(ENDPOINT));
}

#[test]
fn only_https_on_the_approved_host_is_allowed() {
    for bad in [
        "http://lrclib.net/api/get",
        "https://lrclib.net.evil.example/api/get",
        "https://lrclib.net@evil.example/api/get",
        "https://evil.example/?https://lrclib.net/api/",
        "HTTPS://lrclib.net/api/get",
        "https://lrclib.net/other",
        "ftp://lrclib.net/api/get",
        "",
    ] {
        assert!(!is_allowed_url(bad), "{bad}");
    }
    assert!(is_allowed_url("https://lrclib.net/api/get?track_name=x"));
}

#[test]
fn user_agent_names_the_app_and_repo() {
    let ua = user_agent();
    assert!(ua.starts_with("Musigraph/"), "{ua}");
    assert!(ua.ends_with(" (+https://github.com/KabirADnan41/musigraph)"));
}

#[test]
fn debug_output_is_redacted() {
    let s = format!(
        "{:?} {:?} {:?}",
        q(),
        judge(&q(), &rec(200.0)),
        ok(200, PLAIN)
    );
    assert!(
        !s.contains("Synthetic") && !s.contains("Artist") && !s.contains("la la"),
        "{s}"
    );
}

// ---- strict matching ----

#[test]
fn duration_must_match_within_two_seconds() {
    for d in [198.0, 200.0, 202.0, 199.1] {
        assert!(matches!(judge(&q(), &rec(d)), Outcome::Synced(_)), "{d}");
    }
    for d in [
        197.99,
        202.01,
        0.0,
        -200.0,
        1e308,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(judge(&q(), &rec(d)), Outcome::Rejected, "{d}");
    }
    let mut r = rec(200.0);
    r.duration = None;
    assert_eq!(judge(&q(), &r), Outcome::Rejected);
}

#[test]
fn instrumental_and_missing_lyrics_are_rejected() {
    let mut r = rec(200.0);
    r.instrumental = Some(true);
    assert_eq!(judge(&q(), &r), Outcome::Rejected);
    let mut r = rec(200.0);
    r.instrumental = None;
    assert!(matches!(judge(&q(), &r), Outcome::Synced(_)));
    let mut r = rec(200.0);
    (r.synced, r.plain) = (None, None);
    assert_eq!(judge(&q(), &r), Outcome::Rejected);
    let mut r = rec(200.0);
    (r.synced, r.plain) = (Some(String::new()), Some(" \n\t\n".into()));
    assert_eq!(judge(&q(), &r), Outcome::Rejected);
}

#[test]
fn synced_lyrics_are_parsed_with_the_bounded_lrc_parser() {
    let Outcome::Synced(l) = judge(&q(), &rec(200.0)) else {
        panic!("expected synced")
    };
    assert_eq!(l.lines.len(), 2);
    assert_eq!(l.lines[1].text, "na na");
    assert!((l.lines[1].start_s - 3.5).abs() < 1e-9);
}

#[test]
fn plain_only_is_a_distinct_unsynced_result() {
    let mut r = rec(200.0);
    r.synced = None;
    assert_eq!(
        judge(&q(), &r),
        Outcome::Unsynced(vec!["la la la".into(), "na na".into()])
    );
    // malformed synced text falls back to the labelled plain panel, never to invented timing
    r.synced = Some("no timestamps here".into());
    assert!(matches!(judge(&q(), &r), Outcome::Unsynced(_)));
    r.plain = None;
    assert_eq!(judge(&q(), &r), Outcome::Rejected);
    // control and spoofing characters are neutralised, line count bounded
    let mut r = rec(200.0);
    r.synced = None;
    r.plain = Some(format!("a\u{202e}b\u{0}c\n{}", "x\n".repeat(10_000)));
    let Outcome::Unsynced(lines) = judge(&q(), &r) else {
        panic!()
    };
    assert_eq!(lines[0], "a b c");
    assert!(lines.len() <= lrc::MAX_LINES);
}

#[test]
fn version_word_mismatch_is_rejected() {
    let with = |title: &str, album: Option<&str>| {
        let mut r = rec(200.0);
        r.track_name = Some(title.into());
        r.album_name = album.map(Into::into);
        judge(&q(), &r)
    };
    for t in [
        "Synthetic Song (Live)",
        "Synthetic Song - Remix",
        "Synthetic Song (Acoustic)",
        "Synthetic Song [Remastered 2011]",
        "Synthetic Song (Extended Mix)",
        "Synthetic Song (Radio Edit)",
        "Synthetic Song (Instrumental)",
        "Synthetic Song (Karaoke Version)",
        "SYNTHETIC SONG LIVE",
    ] {
        assert_eq!(with(t, Some("Test Album")), Outcome::Rejected, "{t}");
    }
    // a live album is a different recording, too
    assert_eq!(
        with("Synthetic Song", Some("Live at Somewhere")),
        Outcome::Rejected
    );
    // no record title: ambiguous
    assert_eq!(
        judge(
            &q(),
            &Record {
                track_name: None,
                ..rec(200.0)
            }
        ),
        Outcome::Rejected
    );
    // substrings are not words; equal version words on both sides match
    assert!(matches!(
        with("Synthetic Song", Some("Alive Oliver")),
        Outcome::Synced(_)
    ));
    let live = Query::new("Live Song (Remastered)", "Test Artist", "", Some(200.0)).unwrap();
    let mut r = rec(200.0);
    (r.track_name, r.album_name) = (Some("live song - remaster".into()), None);
    assert!(matches!(judge(&live, &r), Outcome::Synced(_)));
    r.track_name = Some("Live Song".into());
    assert_eq!(judge(&live, &r), Outcome::Rejected);
}

// ---- status / transport policy ----

#[test]
fn status_codes_are_classified() {
    assert_eq!(evaluate(&q(), ok(404, "")), Outcome::NotFound);
    for s in [300, 301, 302, 303, 307, 308, 399] {
        assert_eq!(evaluate(&q(), ok(s, "")), Outcome::Refused, "{s}");
    }
    for s in [0, 100, 201, 204, 400, 401, 403, 418, 500, 502, 504, 999] {
        assert_eq!(evaluate(&q(), ok(s, &json("200"))), Outcome::Error, "{s}");
    }
    let ra = |status, ra: Option<Duration>| {
        evaluate(
            &q(),
            Ok(HttpResponse {
                status,
                retry_after: ra,
                body: Vec::new(),
            }),
        )
    };
    let s = Duration::from_secs;
    assert_eq!(ra(429, Some(s(30))), Outcome::RetryAfter(s(30)));
    assert_eq!(ra(503, Some(s(30))), Outcome::RetryAfter(s(30)));
    assert_eq!(ra(429, None), Outcome::RetryAfter(s(60)));
    assert_eq!(ra(429, Some(s(0))), Outcome::RetryAfter(s(1)));
    assert_eq!(ra(429, Some(s(u64::MAX))), Outcome::RetryAfter(s(600)));
}

#[test]
fn transport_errors_and_oversize_are_refused_or_errors() {
    use TransportError::*;
    assert_eq!(evaluate(&q(), Err(TooLarge)), Outcome::Refused);
    assert_eq!(evaluate(&q(), Err(NotAllowed)), Outcome::Refused);
    for e in [Cancelled, Timeout, Network] {
        assert_eq!(evaluate(&q(), Err(e)), Outcome::Error);
    }
    let big = format!("{}{}", json("200"), " ".repeat(MAX_BODY_BYTES));
    assert_eq!(evaluate(&q(), ok(200, &big)), Outcome::Refused);
}

#[test]
fn json_depth_check_ignores_brackets_in_strings() {
    assert!(json_depth_ok(r#"{"a":"[[[[[[[[[[[[[[[[[[[[[[\"[[[["}"#));
    assert!(json_depth_ok(&json("200")));
    assert!(!json_depth_ok(&"[".repeat(17)));
    assert!(json_depth_ok(&"[".repeat(16)));
}

#[cfg(windows)]
#[test]
fn json_body_is_parsed_strictly() {
    assert!(matches!(
        evaluate(&q(), ok(200, &json("200.4"))),
        Outcome::Synced(_)
    ));
    assert_eq!(evaluate(&q(), ok(200, &json("203"))), Outcome::Rejected);
    for bad in [
        "",
        "{",
        "[]",
        "null",
        "not json",
        "{\"duration\":NaN}",
        "{\"a\":1}}",
        "\u{feff}",
    ] {
        assert_eq!(evaluate(&q(), ok(200, bad)), Outcome::Error, "{bad}");
    }
    // invalid UTF-8
    let r = Ok(HttpResponse {
        status: 200,
        retry_after: None,
        body: vec![b'{', 0xff, 0xfe, b'}'],
    });
    assert_eq!(evaluate(&q(), r), Outcome::Error);
    // deep nesting never reaches the system parser
    let deep = format!("{{\"x\":{}{}}}", "[".repeat(5000), "]".repeat(5000));
    assert_eq!(evaluate(&q(), ok(200, &deep)), Outcome::Error);
}

#[cfg(windows)]
#[test]
fn json_hostile_values_are_rejected_not_panicking() {
    // wrong types are treated as missing
    assert_eq!(evaluate(&q(), ok(200, &json("\"200\""))), Outcome::Rejected);
    assert_eq!(evaluate(&q(), ok(200, &json("null"))), Outcome::Rejected);
    // huge / overflowing numbers
    for d in ["1e308", "-1e308", "1e999", "-0", "99999999999999999999999"] {
        let out = evaluate(&q(), ok(200, &json(d)));
        assert!(matches!(out, Outcome::Rejected | Outcome::Error), "{d}");
    }
    // lone surrogate and control escapes inside lyrics: no panic, text sanitised
    let body = r#"{"trackName":"Synthetic Song","albumName":"Test Album","duration":200,"instrumental":false,"plainLyrics":"a\ud800b\u0000c\u202Ed","syncedLyrics":null}"#;
    let Outcome::Unsynced(lines) = evaluate(&q(), ok(200, body)) else {
        panic!("expected unsynced")
    };
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with('a') && lines[0].ends_with('d'));
    assert!(!lines[0].contains('\0') && !lines[0].contains('\u{202e}'));
    // instrumental record
    let body = r#"{"trackName":"Synthetic Song","albumName":"Test Album","duration":200,"instrumental":true,"plainLyrics":null,"syncedLyrics":null}"#;
    assert_eq!(evaluate(&q(), ok(200, body)), Outcome::Rejected);
}

// ---- requester state machine ----

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn on(t0: Instant) -> Requester {
    let mut r = Requester::default();
    r.set_consent(true);
    r.observe(t0, Some(&q()));
    r
}

#[test]
fn no_consent_never_requests() {
    let t0 = Instant::now();
    let mut r = Requester::default();
    let mut t = t0;
    for i in 0..2000u32 {
        // tracks change, repeat, disappear; time jumps small and large
        let track = match i % 7 {
            0 => None,
            1 | 2 => Some(q()),
            _ => Some(qn(i % 13)),
        };
        r.observe(t, track.as_ref());
        t += ms(u64::from(i % 5) * 700 + 1);
        assert_eq!(r.poll(t), None, "step {i}");
        if i % 97 == 0 {
            r.set_consent(false);
        }
        if i % 251 == 0 {
            t += Duration::from_secs(3600);
        }
    }
    // a stray reply while never consented is ignored
    let req = Request {
        query: q(),
        generation: 0,
    };
    assert_eq!(r.on_outcome(t, &req, Outcome::NotFound), None);
    assert_eq!(r.poll(t + Duration::from_secs(10)), None);
}

#[test]
fn revoke_during_debounce_or_flight_stops_everything() {
    let t0 = Instant::now();
    let mut r = on(t0);
    r.set_consent(false);
    for k in 0..100 {
        assert_eq!(r.poll(t0 + ms(1500 + k * 1000)), None);
    }
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    r.set_consent(false);
    assert_ne!(r.generation(), req.generation);
    assert_eq!(r.on_outcome(t0 + ms(2000), &req, Outcome::NotFound), None);
    assert_eq!(r.poll(t0 + ms(9000)), None);
}

#[test]
fn debounce_waits_one_and_a_half_seconds() {
    let t0 = Instant::now();
    let mut r = on(t0);
    assert_eq!(r.poll(t0), None);
    assert_eq!(r.poll(t0 + ms(1499)), None);
    let req = r.poll(t0 + ms(1500)).unwrap();
    assert_eq!(req.query, q());
    assert_eq!(req.generation, r.generation());
}

#[test]
fn skipping_through_tracks_sends_nothing_until_settled() {
    let t0 = Instant::now();
    let mut r = Requester::default();
    r.set_consent(true);
    let mut sent = Vec::new();
    for i in 0..20u64 {
        let t = t0 + ms(i * 1000);
        r.observe(t, Some(&qn(i as u32)));
        sent.extend(r.poll(t + ms(999)));
    }
    assert!(sent.is_empty());
    let last = t0 + ms(19_000);
    let req = r.poll(last + DEBOUNCE).unwrap();
    assert_eq!(req.query, qn(19));
}

#[test]
fn one_in_flight_and_dedupe_with_negative_cache() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    assert_eq!(r.poll(t0 + ms(5000)), None, "one in flight");
    assert_eq!(
        r.on_outcome(t0 + ms(5000), &req, Outcome::NotFound),
        Some(Outcome::NotFound)
    );
    // same track again, also after switching away and back: never re-requested this session
    assert_eq!(r.poll(t0 + ms(60_000)), None);
    r.observe(t0 + ms(61_000), Some(&qn(1)));
    r.observe(t0 + ms(70_000), Some(&q()));
    assert_eq!(r.poll(t0 + ms(80_000)), None);
    for o in [Outcome::Rejected, Outcome::Refused, Outcome::Error] {
        let mut r = on(t0);
        let req = r.poll(t0 + DEBOUNCE).unwrap();
        assert_eq!(r.on_outcome(t0 + DEBOUNCE, &req, o.clone()), Some(o));
        assert_eq!(r.poll(t0 + Duration::from_secs(3600)), None);
    }
}

#[test]
fn positive_reply_for_a_track_no_longer_playing_is_dropped_and_not_cached() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    r.observe(t0 + ms(1600), Some(&qn(7)));
    let lyr = judge(&q(), &rec(200.0));
    assert_eq!(r.on_outcome(t0 + ms(1700), &req, lyr), None);
    // the new track is requested; coming back to the first one asks again
    let req2 = r.poll(t0 + ms(3200)).unwrap();
    assert_eq!(req2.query, qn(7));
    r.on_outcome(t0 + ms(3300), &req2, Outcome::NotFound);
    r.observe(t0 + ms(3400), Some(&q()));
    assert_eq!(r.poll(t0 + ms(4900)).unwrap().query, q());
}

#[test]
fn global_rate_cap_holds_per_minute() {
    let t0 = Instant::now();
    let mut r = Requester::default();
    r.set_consent(true);
    let mut times = Vec::new();
    let mut t = t0;
    for i in 0..200u32 {
        r.observe(t, Some(&qn(i)));
        t += DEBOUNCE;
        if let Some(req) = r.poll(t) {
            times.push(t);
            r.on_outcome(t, &req, Outcome::NotFound);
        }
        t += ms(10);
    }
    assert!(times.len() > MAX_REQUESTS_PER_MINUTE, "cap must recover");
    for (i, a) in times.iter().enumerate() {
        let in_window = times[i..]
            .iter()
            .take_while(|b| b.duration_since(*a) < Duration::from_secs(60))
            .count();
        assert!(in_window <= MAX_REQUESTS_PER_MINUTE, "{in_window}");
    }
}

#[test]
fn retry_after_pauses_and_retries_the_same_track_once() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    let t1 = t0 + ms(2000);
    let pause = Duration::from_secs(30);
    assert_eq!(r.on_outcome(t1, &req, Outcome::RetryAfter(pause)), None);
    assert_eq!(r.poll(t1 + pause), None, "margin added");
    // a new track during the pause waits too
    r.observe(t1 + ms(100), Some(&qn(3)));
    assert_eq!(r.poll(t1 + ms(20_000)), None);
    r.observe(t1 + ms(20_000), Some(&q()));
    let again = r.poll(t1 + pause + Duration::from_secs(1)).unwrap();
    assert_eq!(again.query, q());
}

#[test]
fn errors_back_off_exponentially() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    let t1 = t0 + ms(2000);
    r.on_outcome(t1, &req, Outcome::Error);
    r.observe(t1, Some(&qn(1)));
    assert_eq!(r.poll(t1 + ms(9_999)), None);
    let req = r.poll(t1 + ms(10_000)).unwrap();
    let t2 = t1 + ms(10_000);
    r.on_outcome(t2, &req, Outcome::Error);
    r.observe(t2, Some(&qn(2)));
    assert_eq!(r.poll(t2 + ms(19_999)), None);
    let req = r.poll(t2 + ms(20_000)).unwrap();
    // success resets the backoff
    let t3 = t2 + ms(20_000);
    r.on_outcome(t3, &req, Outcome::NotFound);
    r.on_outcome(t3, &req, Outcome::Error); // duplicate reply: ignored
    r.observe(t3, Some(&qn(3)));
    assert!(r.poll(t3 + DEBOUNCE).is_some());
}

#[test]
fn revoke_invalidates_late_replies_and_clears_memory() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    r.on_outcome(t0 + ms(1600), &req, Outcome::NotFound);
    let mut r2 = on(t0);
    let inflight = r2.poll(t0 + DEBOUNCE).unwrap();
    for r in [&mut r, &mut r2] {
        r.set_consent(false);
        r.set_consent(true);
        assert!(r.current.is_none() && r.settled.is_empty() && r.in_flight.is_none());
        // nothing until the track is observed again and debounced
        assert_eq!(r.poll(t0 + ms(5000)), None);
        r.observe(t0 + ms(5000), Some(&q()));
        assert_eq!(r.poll(t0 + ms(6000)), None);
    }
    // the late reply from before the revoke is discarded and does not block the new request
    let fresh = r2.poll(t0 + ms(6500)).unwrap();
    let lyr = judge(&q(), &rec(200.0));
    assert_eq!(r2.on_outcome(t0 + ms(6600), &inflight, lyr.clone()), None);
    assert_eq!(r2.on_outcome(t0 + ms(6600), &fresh, lyr.clone()), Some(lyr));
    // previously not-found track is asked again after re-consent (memory was cleared)
    assert!(r.poll(t0 + ms(6500)).is_some());
}

#[test]
fn submit_failure_releases_the_slot() {
    let t0 = Instant::now();
    let mut r = on(t0);
    let req = r.poll(t0 + DEBOUNCE).unwrap();
    r.on_submit_failed(&req);
    assert_eq!(r.poll(t0 + ms(1600)).unwrap(), req);
}
