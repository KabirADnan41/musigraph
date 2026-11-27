//! Bounded wait for WinRT async operations (a stalled operation must never hang a worker thread).

/// Waits for a WinRT operation with a hard timeout; yields `Option<T>`. AsyncStatus raw values: 0 started,
/// 1 completed. (`windows_future` is not re-exported by windows-rs here, so `.get()` is unavailable.)
macro_rules! wait {
    ($op:expr) => {{
        match $op.ok() {
            None => None,
            Some(op) => {
                let deadline = ::std::time::Instant::now() + ::std::time::Duration::from_secs(3);
                loop {
                    match op.Status().map(|s| s.0) {
                        Ok(1) => break op.GetResults().ok(),
                        Ok(0) if ::std::time::Instant::now() < deadline => {
                            ::std::thread::sleep(::std::time::Duration::from_millis(5))
                        }
                        Ok(0) => {
                            let _ = op.Cancel();
                            break None;
                        }
                        _ => break None,
                    }
                }
            }
        }
    }};
}
