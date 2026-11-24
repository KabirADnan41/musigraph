//! Pure audio analysis. Contains no OS code and no `unsafe`.
#![forbid(unsafe_code)]

/// Crate identity, used by the workspace smoke test.
pub const NAME: &str = "mg-dsp";

#[cfg(test)]
mod tests {
    #[test]
    fn name_is_stable() {
        assert_eq!(super::NAME, "mg-dsp");
    }
}
