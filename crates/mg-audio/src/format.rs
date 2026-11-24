//! Pure, safe audio-format validation and conversion to mono `f32` (no OS access, no `unsafe`).
//!
//! Trust boundary: the numbers describing a stream come from the audio driver, so they are validated
//! (plan section 13.5) before any buffer is interpreted; samples are sanitised (NaN/inf -> 0, clamp to
//! [-1, 1]) at the conversion boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    F32,
    Pcm16,
    Pcm24,
    Pcm32,
}

impl SampleKind {
    pub fn bytes(self) -> usize {
        match self {
            SampleKind::Pcm16 => 2,
            SampleKind::Pcm24 => 3,
            SampleKind::F32 | SampleKind::Pcm32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub kind: SampleKind,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FormatError {
    Channels(u16),
    SampleRate(u32),
    Unsupported { bits: u16, float: bool },
    BlockAlign { expected: usize, got: u16 },
    PartialFrame { len: usize, block_align: usize },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Channels(c) => write!(f, "unsupported channel count {c}"),
            FormatError::SampleRate(r) => write!(f, "unsupported sample rate {r}"),
            FormatError::Unsupported { bits, float } => {
                write!(f, "unsupported sample format ({bits}-bit, float={float})")
            }
            FormatError::BlockAlign { expected, got } => {
                write!(f, "block align {got} does not match expected {expected}")
            }
            FormatError::PartialFrame { len, block_align } => {
                write!(
                    f,
                    "buffer of {len} bytes is not a whole number of {block_align}-byte frames"
                )
            }
        }
    }
}

impl StreamFormat {
    /// Validates driver-reported values. `bits` is the container size per sample.
    pub fn validate(
        sample_rate: u32,
        channels: u16,
        bits: u16,
        float: bool,
        block_align: u16,
    ) -> Result<Self, FormatError> {
        if !(1..=8).contains(&channels) {
            return Err(FormatError::Channels(channels));
        }
        if !(8_000..=384_000).contains(&sample_rate) {
            return Err(FormatError::SampleRate(sample_rate));
        }
        let kind = match (float, bits) {
            (true, 32) => SampleKind::F32,
            (false, 16) => SampleKind::Pcm16,
            (false, 24) => SampleKind::Pcm24,
            (false, 32) => SampleKind::Pcm32,
            _ => return Err(FormatError::Unsupported { bits, float }),
        };
        let expected = usize::from(channels) * kind.bytes();
        if usize::from(block_align) != expected {
            return Err(FormatError::BlockAlign {
                expected,
                got: block_align,
            });
        }
        Ok(Self {
            sample_rate,
            channels,
            kind,
        })
    }

    pub fn block_align(&self) -> usize {
        usize::from(self.channels) * self.kind.bytes()
    }

    /// Appends the mono (channel average) conversion of `bytes` to `out`; returns the frame count.
    /// `bytes` must hold a whole number of frames (checked, never sliced out of range).
    pub fn to_mono(&self, bytes: &[u8], out: &mut Vec<f32>) -> Result<usize, FormatError> {
        let ba = self.block_align();
        if !bytes.len().is_multiple_of(ba) {
            return Err(FormatError::PartialFrame {
                len: bytes.len(),
                block_align: ba,
            });
        }
        let (ch, w) = (usize::from(self.channels), self.kind.bytes());
        let frames = bytes.len() / ba;
        out.reserve(frames);
        for frame in bytes.chunks_exact(ba) {
            let mut sum = 0.0f32;
            for s in frame.chunks_exact(w) {
                sum += self.sample(s);
            }
            out.push((sum / ch as f32).clamp(-1.0, 1.0));
        }
        Ok(frames)
    }

    fn sample(&self, s: &[u8]) -> f32 {
        let v = match self.kind {
            SampleKind::F32 => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            SampleKind::Pcm16 => f32::from(i16::from_le_bytes([s[0], s[1]])) / 32_768.0,
            SampleKind::Pcm24 => {
                // Sign-extend the 24-bit little-endian value through the top of an i32.
                let raw = i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8;
                raw as f32 / 8_388_608.0
            }
            SampleKind::Pcm32 => {
                (i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0) as f32
            }
        };
        // NaN, inf and denormals (slow for the FFT downstream) become 0; the rest is clamped.
        if v.is_finite() && v.abs() >= f32::MIN_POSITIVE {
            v.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(ch: u16, kind: SampleKind) -> StreamFormat {
        StreamFormat {
            sample_rate: 48_000,
            channels: ch,
            kind,
        }
    }

    #[test]
    fn validates_driver_values() {
        assert!(StreamFormat::validate(48_000, 2, 32, true, 8).is_ok());
        assert!(StreamFormat::validate(44_100, 2, 16, false, 4).is_ok());
        assert!(StreamFormat::validate(96_000, 6, 24, false, 18).is_ok());
        assert_eq!(
            StreamFormat::validate(48_000, 0, 16, false, 0),
            Err(FormatError::Channels(0))
        );
        assert_eq!(
            StreamFormat::validate(48_000, 9, 16, false, 18),
            Err(FormatError::Channels(9))
        );
        assert_eq!(
            StreamFormat::validate(1_000, 2, 16, false, 4),
            Err(FormatError::SampleRate(1_000))
        );
        assert_eq!(
            StreamFormat::validate(1_000_000, 2, 16, false, 4),
            Err(FormatError::SampleRate(1_000_000))
        );
        assert_eq!(
            StreamFormat::validate(48_000, 2, 8, false, 2),
            Err(FormatError::Unsupported {
                bits: 8,
                float: false
            })
        );
        assert_eq!(
            StreamFormat::validate(48_000, 2, 16, true, 4),
            Err(FormatError::Unsupported {
                bits: 16,
                float: true
            })
        );
        assert_eq!(
            StreamFormat::validate(48_000, 2, 16, false, 5),
            Err(FormatError::BlockAlign {
                expected: 4,
                got: 5
            })
        );
    }

    #[test]
    fn converts_each_sample_kind_and_downmixes() {
        let mut out = Vec::new();
        let f = fmt(2, SampleKind::F32);
        let b: Vec<u8> = [0.5f32, -0.25, 1.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f.to_mono(&b, &mut out), Ok(2));
        assert_eq!(out, vec![0.125, 1.0]);

        out.clear();
        let f = fmt(1, SampleKind::Pcm16);
        let b: Vec<u8> = [16_384i16, -32_768, 32_767]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        f.to_mono(&b, &mut out).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6 && out[1] == -1.0 && (out[2] - 1.0).abs() < 1e-3);

        out.clear();
        let f = fmt(1, SampleKind::Pcm24);
        // +0.5 (0x400000), -0.5 (0xC00000), most negative (0x800000)
        f.to_mono(
            &[0x00, 0x00, 0x40, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x80],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0.5, -0.5, -1.0]);

        out.clear();
        let f = fmt(1, SampleKind::Pcm32);
        let b: Vec<u8> = [1_073_741_824i32, i32::MIN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        f.to_mono(&b, &mut out).unwrap();
        assert_eq!(out, vec![0.5, -1.0]);
    }

    #[test]
    fn six_channel_downmix_averages_all_channels() {
        let f = fmt(6, SampleKind::F32);
        let b: Vec<u8> = [0.6f32, 0.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = Vec::new();
        f.to_mono(&b, &mut out).unwrap();
        assert!((out[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn hostile_samples_become_finite_and_bounded() {
        let f = fmt(1, SampleKind::F32);
        let b: Vec<u8> = [
            f32::NAN,
            f32::INFINITY,
            -f32::INFINITY,
            7.0,
            -7.0,
            f32::MIN_POSITIVE / 4.0,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        let mut out = Vec::new();
        f.to_mono(&b, &mut out).unwrap();
        assert!(
            out.iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        );
        // NaN, +inf, -inf, 7.0, -7.0, denormal
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn partial_frames_are_rejected_not_sliced() {
        let f = fmt(2, SampleKind::Pcm16);
        let mut out = Vec::new();
        assert_eq!(
            f.to_mono(&[0u8; 7], &mut out),
            Err(FormatError::PartialFrame {
                len: 7,
                block_align: 4
            })
        );
        assert!(out.is_empty());
        assert_eq!(f.to_mono(&[], &mut out), Ok(0));
    }
}
