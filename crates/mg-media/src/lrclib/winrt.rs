//! Windows.Web.Http transport for the lyrics worker thread. HTTPS to the approved host only, no
//! redirects, no cookies, no HTTP cache reads or writes, no credential UI, bounded body, app-side
//! deadline and cancellation. Tests never send a request with it (no network in tests).

use super::{
    HttpResponse, MAX_BODY_BYTES, REQUEST_TIMEOUT, Transport, TransportError, is_allowed_url,
    user_agent,
};
use std::time::{Duration, Instant};
use windows::Foundation::Uri;
use windows::Storage::Streams::{Buffer, DataReader, InputStreamOptions};
use windows::Web::Http::Filters::{
    HttpBaseProtocolFilter, HttpCacheReadBehavior, HttpCacheWriteBehavior, HttpCookieUsageBehavior,
};
use windows::Web::Http::{HttpClient, HttpCompletionOption};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::core::{HSTRING, h};

/// Bytes requested per stream read; the body never grows past `MAX_BODY_BYTES`.
const CHUNK: u32 = 16 * 1024;

/// Polls a WinRT async operation until done, the deadline passes, or `cancelled()`; on the latter two
/// the operation is cancelled. Yields `Result<T, TransportError>`. AsyncStatus: 0 started, 1 completed.
macro_rules! await_op {
    ($op:expr, $deadline:expr, $cancelled:expr) => {{
        match $op {
            Err(_) => Err(TransportError::Network),
            Ok(op) => loop {
                match op.Status().map(|s| s.0) {
                    Ok(1) => break op.GetResults().map_err(|_| TransportError::Network),
                    Ok(0) if $cancelled() => {
                        let _ = op.Cancel();
                        break Err(TransportError::Cancelled);
                    }
                    Ok(0) if Instant::now() >= $deadline => {
                        let _ = op.Cancel();
                        break Err(TransportError::Timeout);
                    }
                    Ok(0) => std::thread::sleep(Duration::from_millis(5)),
                    _ => break Err(TransportError::Network),
                }
            },
        }
    }};
}

/// Balances one successful `CoInitializeEx` on the creating thread.
struct Com;

impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: only constructed after a successful CoInitializeEx on this thread; the transport is
        // created and dropped on the same worker thread.
        #[allow(unsafe_code)]
        unsafe {
            CoUninitialize()
        };
    }
}

pub struct WinRtTransport {
    client: HttpClient,
    _com: Com, // declared last: dropped after `client`
}

/// All privacy/safety settings of the HTTP stack, in one place.
fn filter() -> windows::core::Result<HttpBaseProtocolFilter> {
    let f = HttpBaseProtocolFilter::new()?;
    f.SetAllowAutoRedirect(false)?;
    f.SetAllowUI(false)?;
    f.SetCookieUsageBehavior(HttpCookieUsageBehavior::NoCookies)?;
    let cc = f.CacheControl()?;
    cc.SetReadBehavior(HttpCacheReadBehavior::NoCache)?;
    cc.SetWriteBehavior(HttpCacheWriteBehavior::NoCache)?;
    Ok(f)
}

impl WinRtTransport {
    /// Creates the client on the CURRENT thread (pass `WinRtTransport::new` to `LrclibClient::start`).
    /// No network activity. `None` if COM or the HTTP stack cannot be set up.
    pub fn new() -> Option<Self> {
        // SAFETY: balanced by `Com::drop` on this thread; MTA so blocking polls are allowed.
        #[allow(unsafe_code)]
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_err() {
            return None;
        }
        let com = Com;
        let client = HttpClient::Create(&filter().ok()?).ok()?;
        let ua_set = client
            .DefaultRequestHeaders()
            .ok()?
            .TryAppendWithoutValidation(h!("User-Agent"), &HSTRING::from(user_agent()))
            .ok()?;
        ua_set.then_some(Self { client, _com: com })
    }
}

impl Transport for WinRtTransport {
    fn get(&self, url: &str, cancelled: &dyn Fn() -> bool) -> Result<HttpResponse, TransportError> {
        if !is_allowed_url(url) {
            return Err(TransportError::NotAllowed);
        }
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        let net = |_: windows::core::Error| TransportError::Network;
        let uri = Uri::CreateUri(&HSTRING::from(url)).map_err(net)?;
        let resp = await_op!(
            self.client
                .GetWithOptionAsync(&uri, HttpCompletionOption::ResponseHeadersRead),
            deadline,
            cancelled
        )?;
        let status = resp
            .StatusCode()
            .ok()
            .and_then(|s| u16::try_from(s.0).ok())
            .unwrap_or(0);
        let retry_after = resp
            .Headers()
            .and_then(|h| h.RetryAfter())
            .and_then(|v| v.Delta())
            .and_then(|d| d.Value())
            .ok()
            .map(|t| Duration::from_secs(u64::try_from(t.Duration / 10_000_000).unwrap_or(0)));
        let body = if status == 200 {
            read_body(&resp, deadline, cancelled)
        } else {
            Ok(Vec::new()) // other statuses: body never read
        };
        let _ = resp.Close();
        Ok(HttpResponse {
            status,
            retry_after,
            body: body?,
        })
    }
}

fn read_body(
    resp: &windows::Web::Http::HttpResponseMessage,
    deadline: Instant,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<u8>, TransportError> {
    let net = |_: windows::core::Error| TransportError::Network;
    let content = resp.Content().map_err(net)?;
    if let Ok(len) = content
        .Headers()
        .and_then(|h| h.ContentLength())
        .and_then(|r| r.Value())
    {
        if len > MAX_BODY_BYTES as u64 {
            return Err(TransportError::TooLarge);
        }
    }
    let stream = await_op!(content.ReadAsInputStreamAsync(), deadline, cancelled)?;
    let mut body = Vec::new();
    loop {
        let buf = Buffer::Create(CHUNK).map_err(net)?;
        let got = await_op!(
            stream.ReadAsync(&buf, CHUNK, InputStreamOptions::Partial),
            deadline,
            cancelled
        )?;
        let n = got.Length().map_err(net)? as usize;
        if n == 0 {
            return Ok(body);
        }
        if body.len() + n > MAX_BODY_BYTES {
            return Err(TransportError::TooLarge);
        }
        let start = body.len();
        body.resize(start + n, 0);
        DataReader::FromBuffer(&got)
            .and_then(|r| r.ReadBytes(&mut body[start..]))
            .map_err(net)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Network-free: builds the stack and reads the settings back.
    #[test]
    fn filter_has_the_privacy_settings() {
        let _t = WinRtTransport::new().expect("HTTP stack available");
        let f = filter().unwrap();
        assert!(!f.AllowAutoRedirect().unwrap());
        assert!(!f.AllowUI().unwrap());
        assert_eq!(
            f.CookieUsageBehavior().unwrap(),
            HttpCookieUsageBehavior::NoCookies
        );
        let cc = f.CacheControl().unwrap();
        assert_eq!(cc.ReadBehavior().unwrap(), HttpCacheReadBehavior::NoCache);
        assert_eq!(cc.WriteBehavior().unwrap(), HttpCacheWriteBehavior::NoCache);
    }

    /// Network-free: disallowed URLs are refused before any request object exists.
    #[test]
    fn non_approved_urls_are_refused_without_a_request() {
        let t = WinRtTransport::new().expect("HTTP stack available");
        for url in [
            "http://lrclib.net/api/get?track_name=x",
            "https://example.invalid/api/get",
            "file:///C:/Windows/win.ini",
        ] {
            assert_eq!(t.get(url, &|| false), Err(TransportError::NotAllowed));
        }
    }
}
