//! Bounded artwork decoding through the Windows imaging stack (WinRT `BitmapDecoder`; no new crate).
//!
//! The bytes come from a media session and are UNTRUSTED. Everything is bounded: input size (the caller
//! caps it at `MAX_ART_BYTES`), claimed dimensions, decoded pixels (the decoder scales while decoding so
//! a large image is never expanded at full size) and the wait for each async step. Call this from a
//! worker thread, never the render loop: decoding takes milliseconds to tens of milliseconds.

use crate::{ArtImage, MAX_ART_DECODE_PIXELS, MAX_ART_SIDE};
use windows::Graphics::Imaging::{
    BitmapAlphaMode, BitmapDecoder, BitmapInterpolationMode, BitmapPixelFormat, BitmapTransform,
    ColorManagementMode, ExifOrientationMode,
};
use windows::Storage::Streams::{DataWriter, InMemoryRandomAccessStream};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};

/// Initialises COM (MTA) for the current thread when needed; balanced on drop.
struct ComGuard(bool);

impl ComGuard {
    fn new() -> Self {
        // SAFETY: plain apartment initialisation; paired with CoUninitialize in Drop only on success.
        // RPC_E_CHANGED_MODE (thread already initialised differently) is an Err and is simply not balanced.
        #[allow(unsafe_code)]
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        Self(hr.is_ok())
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.0 {
            // SAFETY: balances the successful CoInitializeEx in `new` on this same thread.
            #[allow(unsafe_code)]
            unsafe {
                CoUninitialize()
            };
        }
    }
}

/// Size to decode at: fits `MAX_ART_SIDE`, keeps the aspect ratio, never upscales, never zero.
pub(crate) fn fit_size(w: u32, h: u32) -> (u32, u32) {
    let m = w.max(h);
    if m <= MAX_ART_SIDE {
        return (w.max(1), h.max(1));
    }
    let s = f64::from(MAX_ART_SIDE) / f64::from(m);
    (
        ((f64::from(w) * s).round() as u32).max(1),
        ((f64::from(h) * s).round() as u32).max(1),
    )
}

/// Decodes compressed artwork (PNG, JPEG, BMP, GIF, ... whatever Windows can decode) into straight RGBA8
/// at most `MAX_ART_SIDE` on a side. `None` for corrupt, oversized or unsupported data.
pub fn decode_art(bytes: &[u8]) -> Option<ArtImage> {
    if bytes.is_empty() || bytes.len() > crate::MAX_ART_BYTES {
        return None;
    }
    let _com = ComGuard::new();
    let stream = InMemoryRandomAccessStream::new().ok()?;
    let writer = DataWriter::CreateDataWriter(&stream).ok()?;
    writer.WriteBytes(bytes).ok()?;
    wait!(writer.StoreAsync())?;
    let _ = writer.DetachStream();
    stream.Seek(0).ok()?;
    let decoder = wait!(BitmapDecoder::CreateAsync(&stream))?;
    let (w, h) = (decoder.PixelWidth().ok()?, decoder.PixelHeight().ok()?);
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > MAX_ART_DECODE_PIXELS {
        return None;
    }
    let (sw, sh) = fit_size(w, h);
    let transform = BitmapTransform::new().ok()?;
    transform.SetScaledWidth(sw).ok()?;
    transform.SetScaledHeight(sh).ok()?;
    transform
        .SetInterpolationMode(BitmapInterpolationMode::Fant)
        .ok()?;
    let provider = wait!(decoder.GetPixelDataTransformedAsync(
        BitmapPixelFormat::Rgba8,
        BitmapAlphaMode::Straight,
        &transform,
        ExifOrientationMode::IgnoreExifOrientation,
        ColorManagementMode::DoNotColorManage,
    ))?;
    let data = provider.DetachPixelData().ok()?;
    let rgba: Vec<u8> = data.to_vec();
    (rgba.len() == sw as usize * sh as usize * 4).then_some(ArtImage {
        width: sw,
        height: sh,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{png_claiming_size, png_rgba};

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Vec<u8> {
        (0..w * h).flat_map(|_| c).collect()
    }

    #[test]
    fn fit_size_bounds_and_keeps_aspect() {
        assert_eq!(fit_size(100, 50), (100, 50));
        assert_eq!(fit_size(4000, 2000), (512, 256));
        assert_eq!(fit_size(1, 100_000), (1, 512));
        assert_eq!(fit_size(0, 0), (1, 1));
    }

    #[test]
    fn decodes_a_synthetic_png_to_the_right_pixels() {
        let img = decode_art(&png_rgba(8, 4, &solid(8, 4, [200, 30, 60, 255]))).expect("decodes");
        assert_eq!((img.width, img.height), (8, 4));
        assert_eq!(&img.rgba[..4], &[200, 30, 60, 255]);
        assert_eq!(img.rgba.len(), 8 * 4 * 4);
    }

    #[test]
    fn large_images_are_scaled_while_decoding() {
        let img = decode_art(&png_rgba(1024, 512, &solid(1024, 512, [10, 200, 90, 255]))).unwrap();
        assert_eq!((img.width, img.height), (512, 256));
        let px = &img.rgba[..4];
        assert!(
            px[0] < 20 && px[1] > 190 && px[2] > 80 && px[2] < 100,
            "{px:?}"
        );
    }

    #[test]
    fn corrupt_empty_oversized_and_hostile_inputs_are_refused() {
        assert!(decode_art(&[]).is_none());
        assert!(decode_art(b"not an image at all").is_none());
        let mut p = png_rgba(8, 8, &solid(8, 8, [1, 2, 3, 255]));
        let n = p.len();
        p.truncate(n / 2);
        // the imaging stack may return the part it could decode; either way the result stays bounded
        if let Some(i) = decode_art(&p) {
            assert!(
                i.width <= 8 && i.height <= 8 && i.rgba.len() == (i.width * i.height * 4) as usize
            );
        }
        assert!(
            decode_art(&png_claiming_size(100_000, 100_000)).is_none(),
            "claims 10 Gpx"
        );
        assert!(decode_art(&vec![0u8; crate::MAX_ART_BYTES + 1]).is_none());
    }
}
