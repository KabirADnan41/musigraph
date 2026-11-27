//! Windows-only WASAPI loopback capture of the DEFAULT RENDER endpoint ("what you hear").
//!
//! Safety/Privacy contract (plan section 13, ADR-0003):
//! * Only `eRender` + `AUDCLNT_STREAMFLAGS_LOOPBACK` is ever opened. No capture (microphone) endpoint.
//! * Samples are converted to mono f32 and handed to the callback only; never stored, logged or written.
//! * COM is initialised once per capture thread by an RAII guard declared first and dropped last; every
//!   COM object lives in `run_session`, which returns before the guard drops.
//! * `GetBuffer` slices never outlive `ReleaseBuffer`, which runs on every path (RAII `Packet`).
//! * Driver-supplied sizes are validated with checked arithmetic before any slice is built.
//! * No unwinding crosses FFI: the thread entry uses `catch_unwind`; a panic ends capture with an error status.
//! * Default-device changes are detected by polling the endpoint id (no COM callbacks, so no re-entrancy).

use crate::format::StreamFormat;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_LOOPBACK, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator,
    MMDeviceEnumerator, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eConsole, eRender,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::core::GUID;

const POLL: Duration = Duration::from_millis(10);
/// Loopback delivers no packets while nothing plays; after this long we synthesise zero blocks.
const SILENCE_AFTER: Duration = Duration::from_millis(50);
/// Never synthesise more than this much silence per block (no flooding after a stall).
const MAX_SILENCE_BLOCK: Duration = Duration::from_millis(100);
const DEVICE_CHECK: Duration = Duration::from_secs(1);
const FORMAT_TAG_EXTENSIBLE: u16 = 0xFFFE;
const FORMAT_TAG_FLOAT: u16 = 3;
const FORMAT_TAG_PCM: u16 = 1;
const SUBTYPE_PCM: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);
const SUBTYPE_FLOAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);
const MAX_PACKET_BYTES: usize = 1 << 24;

/// Snapshot of the capture thread, for diagnostics and the engine's fallback decision.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub running: bool,
    pub sample_rate: u32,
    pub channels: u16,
    pub opens: u32,
    pub discontinuities: u64,
    /// Capture-loop iterations that had to synthesise silence (an iteration count, NOT a duration).
    pub synthesized_silence_blocks: u64,
    /// Total duration of synthesised silence in milliseconds (the meaningful figure).
    pub synthesized_silence_ms: u64,
    /// Real packets delivered by the driver and the audio time they carried in milliseconds.
    pub real_packets: u64,
    pub real_audio_ms: u64,
    pub last_error: Option<String>,
}

/// Whole milliseconds spanned by `frames` at `sample_rate` (0 for an invalid rate).
fn frames_to_ms(frames: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    frames.saturating_mul(1000) / u64::from(sample_rate)
}

/// Handle to a running loopback capture thread; dropping it stops and joins the thread.
pub struct Loopback {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
    thread: Option<JoinHandle<()>>,
    /// Set by the capture thread as its very last action (even after a panic).
    done: Arc<AtomicBool>,
}

/// How long `stop` waits for the capture thread before detaching it (e.g. a stalled audio driver).
const STOP_TIMEOUT: Duration = Duration::from_secs(3);

struct ComGuard;
impl ComGuard {
    fn new() -> Result<Self, String> {
        // SAFETY: balanced by CoUninitialize in Drop on the same thread.
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
            .ok()
            .map_err(|e| format!("CoInitializeEx failed: {e}"))?;
        Ok(Self)
    }
}
impl Drop for ComGuard {
    fn drop(&mut self) {
        // SAFETY: paired with the successful CoInitializeEx in `new`.
        unsafe { CoUninitialize() };
    }
}

/// Owns the CoTaskMem allocation returned by `GetMixFormat`.
struct MixFormat(*mut WAVEFORMATEX);
impl Drop for MixFormat {
    fn drop(&mut self) {
        // SAFETY: pointer came from GetMixFormat (CoTaskMemAlloc) and is freed exactly once.
        unsafe { CoTaskMemFree(Some(self.0 as *const _)) };
    }
}

/// One `GetBuffer`..`ReleaseBuffer` pair; the release happens in Drop on every path.
struct Packet<'a> {
    client: &'a IAudioCaptureClient,
    frames: u32,
    data: *const u8,
    flags: u32,
}
impl Drop for Packet<'_> {
    fn drop(&mut self) {
        // SAFETY: matches the successful GetBuffer that created this packet.
        let _ = unsafe { self.client.ReleaseBuffer(self.frames) };
    }
}

fn parse_format(mix: &MixFormat) -> Result<StreamFormat, String> {
    // SAFETY: GetMixFormat returned a valid WAVEFORMATEX; the struct is packed so fields are copied unaligned.
    let base = unsafe { std::ptr::read_unaligned(mix.0) };
    let (tag, channels, rate, align, bits) = (
        base.wFormatTag,
        base.nChannels,
        base.nSamplesPerSec,
        base.nBlockAlign,
        base.wBitsPerSample,
    );
    let float = match tag {
        FORMAT_TAG_FLOAT => true,
        FORMAT_TAG_PCM => false,
        FORMAT_TAG_EXTENSIBLE => {
            if usize::from(base.cbSize) < 22 {
                return Err("extensible format with a truncated header".into());
            }
            // SAFETY: cbSize >= 22 means the allocation holds a full WAVEFORMATEXTENSIBLE.
            let ext = unsafe { std::ptr::read_unaligned(mix.0 as *const WAVEFORMATEXTENSIBLE) };
            let sub = ext.SubFormat;
            if sub == SUBTYPE_FLOAT {
                true
            } else if sub == SUBTYPE_PCM {
                false
            } else {
                return Err("unsupported extensible sub-format".into());
            }
        }
        other => return Err(format!("unsupported format tag {other}")),
    };
    StreamFormat::validate(rate, channels, bits, float, align).map_err(|e| e.to_string())
}

fn default_render_id(enumerator: &IMMDeviceEnumerator) -> Result<String, String> {
    // SAFETY: plain COM calls; the PWSTR is freed with CoTaskMemFree after copying.
    unsafe {
        let dev = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(|e| format!("no default render endpoint: {e}"))?;
        let id = dev.GetId().map_err(|e| format!("GetId failed: {e}"))?;
        let s = id.to_string().map_err(|e| format!("bad endpoint id: {e}"));
        CoTaskMemFree(Some(id.0 as *const _));
        s
    }
}

fn now_ns(epoch: Instant) -> u64 {
    epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

type Callback = dyn FnMut(&[f32], u32, u64) + Send;

/// Opens the default render endpoint in loopback and runs until `stop` is set, the default device
/// changes (returns Ok so the caller re-opens) or an error occurs.
fn run_session(
    stop: &AtomicBool,
    status: &Mutex<Status>,
    epoch: Instant,
    on_block: &mut Callback,
) -> Result<(), String> {
    // SAFETY: standard COM activation sequence; every interface is a smart pointer released on drop.
    let (enumerator, client, capture, format, device_id) = unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| format!("device enumerator: {e}"))?;
        let device_id = default_render_id(&enumerator)?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(|e| format!("default endpoint: {e}"))?;
        let client: IAudioClient = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| format!("activate: {e}"))?;
        let mix = MixFormat(
            client
                .GetMixFormat()
                .map_err(|e| format!("GetMixFormat: {e}"))?,
        );
        let format = parse_format(&mix)?;
        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                1_000_000,
                0,
                mix.0,
                None,
            )
            .map_err(|e| format!("Initialize(loopback): {e}"))?;
        let capture: IAudioCaptureClient = client
            .GetService()
            .map_err(|e| format!("GetService: {e}"))?;
        client.Start().map_err(|e| format!("Start: {e}"))?;
        (enumerator, client, capture, format, device_id)
    };
    // Durations accumulate in frames for this open (one sample rate) and convert once per update, so per-packet
    // truncation does not add up; earlier opens' totals are the base.
    let (mut real_base_ms, mut silence_base_ms) = (0, 0);
    let (mut real_frames, mut silence_frames) = (0u64, 0u64);
    if let Ok(mut s) = status.lock() {
        s.sample_rate = format.sample_rate;
        s.channels = format.channels;
        s.opens += 1;
        s.last_error = None;
        (real_base_ms, silence_base_ms) = (s.real_audio_ms, s.synthesized_silence_ms);
    }

    let mut mono: Vec<f32> = Vec::with_capacity(4096);
    let mut last_data = Instant::now();
    let mut last_emit = Instant::now();
    let mut last_check = Instant::now();
    let result = loop {
        if stop.load(Ordering::Acquire) {
            break Ok(());
        }
        // Drain every pending packet.
        loop {
            // SAFETY: valid, started capture client.
            let pending = match unsafe { capture.GetNextPacketSize() } {
                Ok(n) => n,
                Err(e) => return finish(&client, Err(format!("GetNextPacketSize: {e}"))),
            };
            if pending == 0 {
                break;
            }
            let (mut data, mut frames, mut flags) = (std::ptr::null_mut::<u8>(), 0u32, 0u32);
            // SAFETY: out-pointers are valid locals.
            let hr = unsafe { capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None) };
            if let Err(e) = hr {
                return finish(&client, Err(format!("GetBuffer: {e}")));
            }
            let packet = Packet {
                client: &capture,
                frames,
                data,
                flags,
            };
            if packet.frames == 0 {
                break; // AUDCLNT_S_BUFFER_EMPTY; Packet::drop releases
            }
            mono.clear();
            let bytes = (packet.frames as usize).checked_mul(format.block_align());
            match bytes {
                Some(n) if n <= MAX_PACKET_BYTES => {
                    if packet.flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0 {
                        mono.resize(packet.frames as usize, 0.0);
                    } else {
                        if packet.data.is_null() {
                            return finish(&client, Err("driver returned a null buffer".into()));
                        }
                        // SAFETY: GetBuffer promised `frames` frames of block_align bytes at `data`, valid until
                        // ReleaseBuffer (Packet::drop, after this slice's last use).
                        let slice = unsafe { std::slice::from_raw_parts(packet.data, n) };
                        if let Err(e) = format.to_mono(slice, &mut mono) {
                            return finish(&client, Err(e.to_string()));
                        }
                    }
                    if let Ok(mut s) = status.lock() {
                        s.real_packets += 1;
                        real_frames = real_frames.saturating_add(u64::from(packet.frames));
                        s.real_audio_ms = real_base_ms
                            .saturating_add(frames_to_ms(real_frames, format.sample_rate));
                        if packet.flags & (AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32) != 0 {
                            s.discontinuities += 1;
                        }
                    }
                }
                _ => {
                    return finish(
                        &client,
                        Err("driver reported an invalid packet size".into()),
                    );
                }
            }
            drop(packet);
            if !mono.is_empty() {
                on_block(&mono, format.sample_rate, now_ns(epoch));
            }
            last_data = Instant::now();
            last_emit = last_data;
        }
        // Silence: loopback sends nothing while nothing plays, so keep the visual ticking with zeros.
        let n = silence_block_len(last_data.elapsed(), last_emit.elapsed(), format.sample_rate);
        if n > 0 {
            mono.clear();
            mono.resize(n, 0.0);
            on_block(&mono, format.sample_rate, now_ns(epoch));
            last_emit = Instant::now();
            if let Ok(mut s) = status.lock() {
                s.synthesized_silence_blocks += 1;
                silence_frames = silence_frames.saturating_add(n as u64);
                s.synthesized_silence_ms = silence_base_ms
                    .saturating_add(frames_to_ms(silence_frames, format.sample_rate));
            }
        }
        // Default device changed (or vanished)? Return so the supervisor re-opens.
        if last_check.elapsed() >= DEVICE_CHECK {
            last_check = Instant::now();
            match default_render_id(&enumerator) {
                Ok(id) if id == device_id => {}
                _ => break Ok(()),
            }
        }
        std::thread::sleep(POLL);
    };
    finish(&client, result)
}

/// Number of zero samples to synthesise now: none while real packets are still arriving (idle shorter
/// than `SILENCE_AFTER`), otherwise the time since the last emitted block, capped at `MAX_SILENCE_BLOCK`.
fn silence_block_len(idle: Duration, since_emit: Duration, sample_rate: u32) -> usize {
    if idle < SILENCE_AFTER {
        return 0;
    }
    (since_emit.min(MAX_SILENCE_BLOCK).as_secs_f64() * f64::from(sample_rate)) as usize
}

fn finish(client: &IAudioClient, r: Result<(), String>) -> Result<(), String> {
    // SAFETY: Stop on a valid client; errors are irrelevant during teardown.
    let _ = unsafe { client.Stop() };
    r
}

fn thread_main(stop: Arc<AtomicBool>, status: Arc<Mutex<Status>>, mut on_block: Box<Callback>) {
    let epoch = Instant::now();
    let _com = match ComGuard::new() {
        Ok(g) => g,
        Err(e) => {
            set_error(&status, e);
            return;
        }
    };
    if let Ok(mut s) = status.lock() {
        s.running = true;
    }
    let mut backoff = Duration::from_millis(250);
    while !stop.load(Ordering::Acquire) {
        match run_session(&stop, &status, epoch, &mut *on_block) {
            Ok(()) => backoff = Duration::from_millis(250),
            Err(e) => {
                set_error(&status, e);
                sleep_interruptible(&stop, backoff);
                backoff = (backoff * 2).min(Duration::from_secs(2));
            }
        }
    }
    if let Ok(mut s) = status.lock() {
        s.running = false;
    }
}

fn sleep_interruptible(stop: &AtomicBool, total: Duration) {
    let end = Instant::now() + total;
    while Instant::now() < end && !stop.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn set_error(status: &Mutex<Status>, e: String) {
    if let Ok(mut s) = status.lock() {
        s.last_error = Some(e);
    }
}

impl Loopback {
    /// Starts capturing. `on_block(mono_samples, sample_rate, t_end_ns)` runs on the capture thread;
    /// `t_end_ns` is a monotonic process-local clock for the newest sample of the block.
    pub fn start(on_block: impl FnMut(&[f32], u32, u64) + Send + 'static) -> Result<Self, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(Status::default()));
        let done = Arc::new(AtomicBool::new(false));
        let (s2, st2, done2) = (stop.clone(), status.clone(), done.clone());
        let thread = std::thread::Builder::new()
            .name("mg-audio-loopback".into())
            .spawn(move || {
                let st3 = st2.clone();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    thread_main(s2, st2, Box::new(on_block))
                }));
                if r.is_err() {
                    set_error(&st3, "capture thread panicked".into());
                    if let Ok(mut s) = st3.lock() {
                        s.running = false;
                    }
                }
                done2.store(true, Ordering::Release);
            })
            .map_err(|e| format!("cannot start capture thread: {e}"))?;
        Ok(Self {
            stop,
            status,
            thread: Some(thread),
            done,
        })
    }

    pub fn status(&self) -> Status {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Stops the capture thread. Joins it if it finishes within `STOP_TIMEOUT`; otherwise (a stalled
    /// driver call) the thread is detached and the failure is recorded, so shutdown can never hang.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let Some(t) = self.thread.take() else { return };
        let deadline = Instant::now() + STOP_TIMEOUT;
        while !self.done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if self.done.load(Ordering::Acquire) {
            let _ = t.join();
        } else {
            set_error(
                &self.status,
                "capture thread did not stop in time; detached".into(),
            );
        }
    }
}

impl Drop for Loopback {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_silence_is_synthesised_while_packets_still_arrive() {
        let ms = Duration::from_millis;
        assert_eq!(silence_block_len(ms(10), ms(10), 48_000), 0);
        assert_eq!(silence_block_len(ms(49), ms(49), 48_000), 0);
    }

    #[test]
    fn frames_to_ms_is_exact_and_safe() {
        assert_eq!(frames_to_ms(48_000, 48_000), 1000);
        assert_eq!(frames_to_ms(441, 44_100), 10);
        assert_eq!(frames_to_ms(5, 0), 0);
        assert_eq!(frames_to_ms(u64::MAX, 48_000), u64::MAX / 48_000);
    }

    #[test]
    fn silence_blocks_follow_the_clock_and_are_capped() {
        let ms = Duration::from_millis;
        assert_eq!(silence_block_len(ms(60), ms(10), 48_000), 480);
        assert_eq!(silence_block_len(ms(60), ms(10), 44_100), 441);
        // after a long stall the block is capped (no flooding)
        assert_eq!(silence_block_len(ms(5_000), ms(5_000), 48_000), 4_800);
        assert_eq!(silence_block_len(ms(60), ms(0), 48_000), 0);
    }

    /// Start/stop lifecycle on the real machine: must never hang or panic whether or not audio plays,
    /// and the capture thread must be joined (not detached) in normal operation.
    #[test]
    fn start_stop_cycles_join_cleanly() {
        for _ in 0..20 {
            let mut cap = Loopback::start(|_, _, _| {}).expect("thread starts");
            std::thread::sleep(Duration::from_millis(30));
            let t0 = Instant::now();
            cap.stop();
            assert!(
                t0.elapsed() < Duration::from_secs(2),
                "stop took {:?}",
                t0.elapsed()
            );
            assert!(cap.thread.is_none() && cap.done.load(Ordering::Acquire));
            let st = cap.status();
            assert!(!st.running);
            assert!(
                !st.last_error
                    .as_deref()
                    .unwrap_or("")
                    .contains("did not stop"),
                "{:?}",
                st.last_error
            );
        }
    }
}
