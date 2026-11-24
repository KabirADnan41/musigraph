//! One-pole attack/release smoother, independent of the frame rate (time-constant based).

/// Smooths a value toward its input: `attack_ms` when rising, `release_ms` when falling.
#[derive(Clone, Copy, Debug)]
pub struct Smoother {
    attack_ms: f32,
    release_ms: f32,
    y: f32,
}

impl Smoother {
    pub fn new(attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_ms: attack_ms.max(0.001),
            release_ms: release_ms.max(0.001),
            y: 0.0,
        }
    }

    pub fn value(&self) -> f32 {
        self.y
    }

    pub fn reset(&mut self) {
        self.y = 0.0;
    }

    /// Advances by `dt_s` seconds toward `x` and returns the new value. Non-finite inputs count as 0.
    pub fn step(&mut self, x: f32, dt_s: f32) -> f32 {
        let x = if x.is_finite() { x } else { 0.0 };
        let tau = if x > self.y {
            self.attack_ms
        } else {
            self.release_ms
        };
        let a = 1.0 - (-(dt_s.max(0.0)) * 1000.0 / tau).exp();
        self.y += (x - self.y) * a;
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_response_reaches_63_percent_at_one_time_constant() {
        let mut s = Smoother::new(100.0, 100.0);
        let mut v = 0.0;
        for _ in 0..100 {
            v = s.step(1.0, 0.001); // 100 ms total
        }
        assert!((v - 0.632).abs() < 0.03, "v = {v}");
    }

    #[test]
    fn attack_is_faster_than_release_when_configured() {
        let mut s = Smoother::new(10.0, 500.0);
        for _ in 0..20 {
            s.step(1.0, 0.005); // 100 ms
        }
        let risen = s.value();
        assert!(risen > 0.99, "risen = {risen}");
        for _ in 0..20 {
            s.step(0.0, 0.005); // 100 ms
        }
        assert!(s.value() > 0.7, "fell too fast: {}", s.value());
    }

    #[test]
    fn independent_of_step_size_and_safe_on_bad_input() {
        let (mut a, mut b) = (Smoother::new(50.0, 50.0), Smoother::new(50.0, 50.0));
        for _ in 0..10 {
            a.step(1.0, 0.01);
        }
        for _ in 0..100 {
            b.step(1.0, 0.001);
        }
        assert!((a.value() - b.value()).abs() < 1e-3);
        assert!(a.step(f32::NAN, 0.01).is_finite());
        assert!(a.step(f32::INFINITY, -1.0).is_finite());
    }
}
