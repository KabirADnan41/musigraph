//! Windows "reduced animation" preference (Settings > Accessibility > Visual effects > Animation effects).
//!
//! Purpose: tell the engine whether the OS wants motion minimised. Read-only: this module never changes
//! the system setting. Input/output: `animations_enabled` queries the OS (`None` when unknown or not
//! Windows); `reduced_motion` combines it with the developer `--reduced-motion` flag (pure).
//! Side effects: one cheap `SystemParametersInfoW` read per call; callers poll about once per second,
//! never per frame.

/// `Some(false)` when the user turned animation effects off, `Some(true)` when on, `None` if the query
/// failed or the platform has no such setting.
#[cfg(windows)]
pub fn animations_enabled() -> Option<bool> {
    use windows::Win32::UI::WindowsAndMessaging::{
        SPI_GETCLIENTAREAANIMATION, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SystemParametersInfoW,
    };
    let mut on: i32 = 0; // BOOL
    // SAFETY: `pvParam` points to a live, writable BOOL (4 bytes) for the duration of the call, which is
    // what SPI_GETCLIENTAREAANIMATION requires; uiParam is 0 and no flags are passed, so nothing is written
    // to the system or broadcast.
    let r = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            Some((&raw mut on).cast()),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
    };
    r.ok().map(|()| on != 0)
}

#[cfg(not(windows))]
pub fn animations_enabled() -> Option<bool> {
    None
}

/// Reduced motion applies when forced by the developer flag or when the OS says animations are off.
/// An unknown OS state never reduces motion.
pub fn reduced_motion(dev_flag: bool) -> bool {
    reduced_from(dev_flag, animations_enabled())
}

fn reduced_from(dev_flag: bool, os: Option<bool>) -> bool {
    dev_flag || os == Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_flag_forces_reduced_motion() {
        for os in [None, Some(true), Some(false)] {
            assert!(reduced_from(true, os), "{os:?}");
        }
    }

    #[test]
    fn unknown_or_enabled_means_full_motion_and_disabled_means_reduced() {
        assert!(!reduced_from(false, None));
        assert!(!reduced_from(false, Some(true)));
        assert!(reduced_from(false, Some(false)));
    }

    #[test]
    fn querying_the_os_does_not_panic_and_is_stable() {
        let a = animations_enabled();
        assert_eq!(a, animations_enabled());
        let _ = reduced_motion(false);
        assert!(reduced_motion(true));
    }
}
