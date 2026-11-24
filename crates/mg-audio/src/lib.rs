//! WASAPI loopback capture. With `mg-gfx`, the only crate allowed to contain `unsafe`
//! (all of it in the Windows-only `capture` module; `format` is pure and safe).
//!
//! Privacy invariant (ADR-0003): only the render endpoint is opened in loopback mode, never a capture
//! (microphone) endpoint; samples are never written to disk or logged, only derived numbers leave.

#![deny(unsafe_code)]

pub mod format;

#[cfg(windows)]
#[allow(unsafe_code)] // the single place with COM/FFI; see the module's safety contract
pub mod capture;

/// Crate identity, used by the workspace smoke test.
pub const NAME: &str = "mg-audio";

#[cfg(test)]
mod tests {
    #[test]
    fn name_is_stable() {
        assert_eq!(super::NAME, "mg-audio");
    }
}
