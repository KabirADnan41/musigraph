//! Worker thread + facade: the engine calls `update` once per frame (cheap, never blocks); the network
//! and JSON work happen on one worker thread fed by a bounded queue. Revoke and shutdown cancel the
//! in-flight request through a shared generation counter / stop flag that the transport polls.

use super::{Outcome, Query, Request, Requester, Transport, evaluate};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long `Drop` waits for the worker before detaching it (a stalled OS call never hangs shutdown).
const JOIN_WAIT: Duration = Duration::from_secs(2);

pub struct LrclibClient {
    req: Requester,
    jobs: Option<SyncSender<Request>>,
    results: Receiver<(Request, Outcome)>,
    live_gen: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// Lyrics for the current track only (RAM; dropped on track change and on revoke).
    lyrics: Option<(Query, Outcome)>,
    sent: u64,
}

impl LrclibClient {
    /// Starts the worker. `make` runs ON the worker thread and the transport never leaves it (it need
    /// not be `Send`; the COM-owning `WinRtTransport` is not), so COM is initialised and released on
    /// that one thread. `None` means no transport: every request then ends as `Outcome::Error`.
    /// Consent starts OFF.
    pub fn start<T, F>(make: F) -> std::io::Result<Self>
    where
        T: Transport,
        F: FnOnce() -> Option<T> + Send + 'static,
    {
        let (jobs, job_rx) = mpsc::sync_channel::<Request>(1);
        let (res_tx, results) = mpsc::channel();
        let live_gen = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let (g, s, d) = (live_gen.clone(), stop.clone(), done.clone());
        let thread = std::thread::Builder::new()
            .name("mg-lyrics-net".into())
            .spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let Some(t) = make() else { return };
                    while let Ok(job) = job_rx.recv() {
                        let cancelled = || {
                            s.load(Ordering::Acquire) || g.load(Ordering::Acquire) != job.generation
                        };
                        if cancelled() {
                            continue;
                        }
                        let outcome = evaluate(&job.query, t.get(&job.query.url(), &cancelled));
                        if res_tx.send((job, outcome)).is_err() {
                            break;
                        }
                    }
                }));
                d.store(true, Ordering::Release);
            })?;
        Ok(Self {
            req: Requester::default(),
            jobs: Some(jobs),
            results,
            live_gen,
            stop,
            done,
            thread: Some(thread),
            lyrics: None,
            sent: 0,
        })
    }

    pub fn consented(&self) -> bool {
        self.req.consented()
    }

    /// Grant or revoke. Revoke cancels the in-flight request, invalidates pending replies and clears all
    /// lyrics and track memory.
    pub fn set_consent(&mut self, on: bool) {
        self.req.set_consent(on);
        self.live_gen
            .store(self.req.generation(), Ordering::Release);
        if !on {
            self.lyrics = None;
            while self.results.try_recv().is_ok() {}
        }
    }

    /// Requests handed to the worker so far (a count; safe to log).
    pub fn requests_sent(&self) -> u64 {
        self.sent
    }

    /// Call every frame with the playing track. Never blocks. Returns the outcome for exactly this track,
    /// once known (`None` while waiting, without consent, or for another track).
    pub fn update(&mut self, now: Instant, track: Option<&Query>) -> Option<&Outcome> {
        loop {
            match self.results.try_recv() {
                Ok((r, o)) => self.deliver(now, r, o),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // worker gone (no transport or panicked): whatever it held fails, nothing hangs
                    if let Some(r) = self.req.in_flight().cloned() {
                        self.deliver(now, r, Outcome::Error);
                    }
                    break;
                }
            }
        }
        self.req.observe(now, track);
        if self.lyrics.as_ref().map(|(q, _)| q) != track {
            self.lyrics = None;
        }
        if let Some(r) = self.req.poll(now) {
            match self.jobs.as_ref().map(|j| j.try_send(r.clone())) {
                Some(Ok(())) => self.sent += 1,
                Some(Err(TrySendError::Full(_))) => self.req.on_submit_failed(&r),
                None | Some(Err(TrySendError::Disconnected(_))) => {
                    self.deliver(now, r, Outcome::Error)
                }
            }
        }
        self.lyrics.as_ref().map(|(_, o)| o)
    }

    fn deliver(&mut self, now: Instant, r: Request, o: Outcome) {
        if let Some(o) = self.req.on_outcome(now, &r, o) {
            self.lyrics = Some((r.query, o));
        }
    }
}

impl Drop for LrclibClient {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.jobs = None; // wakes the worker's recv
        self.lyrics = None;
        let Some(t) = self.thread.take() else { return };
        let deadline = Instant::now() + JOIN_WAIT;
        while !self.done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        if self.done.load(Ordering::Acquire) {
            let _ = t.join();
        } // else detached
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HttpResponse, TransportError};
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Mock {
        calls: Arc<Mutex<Vec<String>>>,
        /// When set, `get` blocks until cancelled and records that it saw the cancellation.
        block: bool,
        saw_cancel: Arc<AtomicBool>,
        started: Arc<AtomicBool>,
        /// Statuses to answer first (front first); 404 once empty.
        script: Arc<Mutex<std::collections::VecDeque<u16>>>,
    }

    impl Transport for Mock {
        fn get(
            &self,
            url: &str,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<HttpResponse, TransportError> {
            self.calls.lock().unwrap().push(url.to_string());
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
            let status = self.script.lock().unwrap().pop_front().unwrap_or(404);
            Ok(HttpResponse {
                status,
                retry_after: None,
                body: Vec::new(),
            })
        }
    }

    fn q() -> Query {
        Query::new("Synthetic Song", "Test Artist", "", Some(180.0)).unwrap()
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

    #[test]
    fn without_consent_the_transport_is_never_called() {
        let m = Mock::default();
        let m2 = m.clone();
        let mut c = LrclibClient::start(move || Some(m2)).unwrap();
        let t0 = Instant::now();
        for i in 0..500u64 {
            assert!(
                c.update(t0 + Duration::from_millis(i * 50), Some(&q()))
                    .is_none()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
        assert!(m.calls.lock().unwrap().is_empty());
        assert_eq!(c.requests_sent(), 0);
    }

    #[test]
    fn consented_lookup_reaches_the_transport_once_and_reports() {
        let m = Mock::default();
        let m2 = m.clone();
        let mut c = LrclibClient::start(move || Some(m2)).unwrap();
        c.set_consent(true);
        let t0 = Instant::now();
        assert!(c.update(t0, Some(&q())).is_none());
        let t1 = t0 + Duration::from_secs(2);
        assert!(wait_until(
            || c.update(t1, Some(&q())) == Some(&Outcome::NotFound)
        ));
        for k in 0..50 {
            c.update(t1 + Duration::from_secs(k), Some(&q()));
        }
        let calls = m.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].starts_with(super::super::ALLOWED_PREFIX));
        // outcome belongs to that track only
        drop(calls);
        let other = Query::new("Other", "Test Artist", "", Some(180.0)).unwrap();
        assert!(c.update(t1, Some(&other)).is_none());
    }

    #[test]
    fn a_server_error_is_retried_once_after_the_backoff_then_reported() {
        let m = Mock::default();
        m.script.lock().unwrap().extend([500, 500, 500]);
        let m2 = m.clone();
        let mut c = LrclibClient::start(move || Some(m2)).unwrap();
        c.set_consent(true);
        let t0 = Instant::now();
        c.update(t0, Some(&q()));
        let t1 = t0 + Duration::from_secs(2);
        c.update(t1, Some(&q()));
        assert!(wait_until(|| m.calls.lock().unwrap().len() == 1));
        // reply is consumed, then the backoff (10 s) holds the retry back
        std::thread::sleep(Duration::from_millis(50));
        assert!(c.update(t1 + Duration::from_secs(5), Some(&q())).is_none());
        assert_eq!(m.calls.lock().unwrap().len(), 1);
        let t2 = t1 + Duration::from_secs(16); // reply was taken in at t1+5, so the pause ends at t1+15
        assert!(wait_until(
            || c.update(t2, Some(&q())) == Some(&Outcome::Error)
        ));
        for k in 0..20 {
            c.update(t2 + Duration::from_secs(400 + k), Some(&q()));
        }
        assert_eq!(m.calls.lock().unwrap().len(), 2, "exactly one retry");
    }

    #[test]
    fn revoke_cancels_in_flight_and_discards_the_reply() {
        let m = Mock {
            block: true,
            ..Mock::default()
        };
        let m2 = m.clone();
        let mut c = LrclibClient::start(move || Some(m2)).unwrap();
        c.set_consent(true);
        let t0 = Instant::now();
        c.update(t0, Some(&q()));
        c.update(t0 + Duration::from_secs(2), Some(&q()));
        assert!(wait_until(|| m.started.load(Ordering::Acquire)));
        c.set_consent(false);
        assert!(wait_until(|| m.saw_cancel.load(Ordering::Acquire)));
        std::thread::sleep(Duration::from_millis(20));
        assert!(c.update(t0 + Duration::from_secs(3), Some(&q())).is_none());
        assert!(c.lyrics.is_none());
    }

    #[test]
    fn drop_mid_request_returns_promptly() {
        let m = Mock {
            block: true,
            ..Mock::default()
        };
        let m2 = m.clone();
        let mut c = LrclibClient::start(move || Some(m2)).unwrap();
        c.set_consent(true);
        let t0 = Instant::now();
        c.update(t0, Some(&q()));
        c.update(t0 + Duration::from_secs(2), Some(&q()));
        assert!(wait_until(|| m.started.load(Ordering::Acquire)));
        let before = Instant::now();
        drop(c);
        assert!(before.elapsed() < Duration::from_secs(1));
        assert!(m.saw_cancel.load(Ordering::Acquire));
    }

    #[test]
    fn missing_transport_degrades_to_error_without_hanging() {
        let mut c = LrclibClient::start(|| None::<Mock>).unwrap();
        c.set_consent(true);
        let t0 = Instant::now();
        c.update(t0, Some(&q()));
        let t1 = t0 + Duration::from_secs(2);
        assert!(wait_until(
            || c.update(t1, Some(&q())) == Some(&Outcome::Error)
        ));
    }
}
