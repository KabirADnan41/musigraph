//! WASAPI loopback capture. With `mg-gfx`, the only crate allowed to contain `unsafe`.

/// Crate identity, used by the workspace smoke test.
pub const NAME: &str = "mg-audio";

#[cfg(test)]
mod tests {
    #[test]
    fn name_is_stable() {
        assert_eq!(super::NAME, "mg-audio");
    }
}
