//! wgpu rendering, frame pacing and scenes.

pub mod context;
pub mod fullscreen;
pub mod report;
pub mod scene;
pub mod stress;

/// Crate identity, used by the workspace smoke test.
pub const NAME: &str = "mg-gfx";

#[cfg(test)]
mod tests {
    #[test]
    fn name_is_stable() {
        assert_eq!(super::NAME, "mg-gfx");
    }
}
