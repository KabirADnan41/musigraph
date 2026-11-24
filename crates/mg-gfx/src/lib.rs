//! wgpu rendering, frame pacing and scenes.

pub mod report;

/// Crate identity, used by the workspace smoke test.
pub const NAME: &str = "mg-gfx";

#[cfg(test)]
mod tests {
    #[test]
    fn name_is_stable() {
        assert_eq!(super::NAME, "mg-gfx");
    }
}
