//! Scene switching state machine (pure). One crossfade at a time: a request made while a fade runs is
//! remembered (latest wins) and starts when the running fade completes, so the picture never pops back to a
//! scene that is only partly visible and never cuts. Audio capture is not involved.

use mg_gfx::scene::SceneKind;

/// Crossfade length in seconds.
pub const XFADE_SECS: f32 = 1.0;

pub struct SceneSwitch {
    shown: SceneKind,
    fade: Option<(SceneKind, f32)>,
    pending: Option<SceneKind>,
}

impl SceneSwitch {
    pub fn new(start: SceneKind) -> Self {
        Self {
            shown: start,
            fade: None,
            pending: None,
        }
    }

    /// The scene that is (or will be) the destination: base for "next/previous".
    pub fn target(&self) -> SceneKind {
        self.pending
            .or(self.fade.map(|(to, _)| to))
            .unwrap_or(self.shown)
    }

    /// Asks for `to`. Returns true if this started a crossfade right now.
    pub fn request(&mut self, to: SceneKind) -> bool {
        match self.fade {
            Some((cur, _)) => {
                // never restart or cut a running fade; remember the latest different destination
                self.pending = (to != cur).then_some(to);
                false
            }
            None if to != self.shown => {
                self.fade = Some((to, 0.0));
                true
            }
            None => false,
        }
    }

    /// Advances by `dt` seconds. Returns (scene to draw, crossfade `(to, eased blend)` if one is running).
    pub fn update(&mut self, dt: f32) -> (SceneKind, Option<(SceneKind, f32)>) {
        if let Some((to, t)) = self.fade {
            let dt = if dt.is_finite() {
                dt.clamp(0.0, 0.25)
            } else {
                0.0
            };
            let t = t + dt / XFADE_SECS;
            if t >= 1.0 {
                self.shown = to;
                self.fade = None;
                if let Some(p) = self.pending.take() {
                    if p != self.shown {
                        self.fade = Some((p, 0.0));
                    }
                }
            } else {
                self.fade = Some((to, t));
            }
        }
        let blend = self.fade.map(|(to, t)| (to, t * t * (3.0 - 2.0 * t)));
        (self.shown, blend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SceneKind::*;

    fn run(s: &mut SceneSwitch, secs: f32) -> (SceneKind, Option<(SceneKind, f32)>) {
        let mut r = (s.shown, None);
        for _ in 0..(secs * 60.0) as usize {
            r = s.update(1.0 / 60.0);
        }
        r
    }

    #[test]
    fn a_switch_fades_over_about_a_second_and_then_shows_the_target_alone() {
        let mut s = SceneSwitch::new(Spectrum);
        assert!(s.request(Aurora));
        let (shown, x) = run(&mut s, 0.5);
        assert_eq!(shown, Spectrum);
        let (to, t) = x.unwrap();
        assert_eq!(to, Aurora);
        assert!((0.3..0.7).contains(&t));
        assert_eq!(run(&mut s, 1.0), (Aurora, None));
    }

    #[test]
    fn requests_during_a_fade_never_restart_or_cut_it() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        let (_, before) = run(&mut s, 0.4);
        // same target again, the original scene, and a third scene: none may disturb the running fade
        for k in [Aurora, Spectrum, Focus] {
            s.request(k);
            let (_, now) = s.update(0.0);
            assert_eq!(
                now.map(|(to, _)| to),
                Some(Aurora),
                "fade target unchanged by {k:?}"
            );
            assert!(
                now.unwrap().1 >= before.unwrap().1,
                "progress never goes back"
            );
        }
        // the latest request (Focus) starts after the first fade completes
        assert_eq!(s.target(), Focus);
        let (shown, x) = run(&mut s, 0.7);
        assert_eq!(shown, Aurora);
        assert_eq!(x.map(|(to, _)| to), Some(Focus));
        assert_eq!(run(&mut s, 1.2), (Focus, None));
    }

    #[test]
    fn asking_for_the_scene_being_faded_to_drops_a_pending_request() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        s.request(Focus);
        s.request(Aurora); // changed their mind: just finish the current fade
        assert_eq!(s.target(), Aurora);
        assert_eq!(run(&mut s, 3.0), (Aurora, None));
    }

    #[test]
    fn requesting_the_scene_already_shown_does_nothing() {
        let mut s = SceneSwitch::new(Focus);
        assert!(!s.request(Focus));
        assert_eq!(s.update(1.0), (Focus, None));
    }

    #[test]
    fn hostile_time_steps_are_harmless() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        for dt in [f32::NAN, -5.0, 1e9, f32::INFINITY, 0.0] {
            let _ = s.update(dt);
        }
        let (_, x) = s.update(0.0);
        assert!(x.is_none_or(|(_, t)| (0.0..=1.0).contains(&t)));
    }
}
