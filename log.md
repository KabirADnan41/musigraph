# Musigraph change log (log-first review, see project instructions)

## 2026-11-25 P0.1 workspace skeleton
- Branch feature/poc-audio-fullscreen. Files: Cargo.toml (workspace, edition 2024, rust-version 1.87, license Apache-2.0, forbid-style lints), rust-toolchain.toml (pins 1.100.0 + clippy/rustfmt), crates/{mg-dsp,mg-audio,mg-gfx,mg-engine} (name-constant smoke tests, engine prints a skeleton line), scripts/verify.ps1 (fmt, clippy -D warnings, test, build, Cargo.lock source check, no-eCapture grep gate, tracked-file allow-list, gitleaks).
- Build output goes to F:\musigraph\builds\target through a machine-local F:\musigraph\.cargo\config.toml OUTSIDE the repo (not committed, so no machine paths are published).
- No third-party dependencies yet (batch 1 is resolved in the next slice). mg-dsp has #![forbid(unsafe_code)]; workspace denies unsafe_op_in_unsafe_fn.
- Decisions: no scene trait, no shared types crate yet (plan v2). Toolchain pinned to the exact version so builds are reproducible; this installs a second 1.100.0 toolchain directory on F:.
- Verification run: scripts\verify.ps1 passes (fmt, clippy, 3 smoke tests, build, lockfile check, privacy gate, allow-list, gitleaks no leaks). Not run: osv-scanner (not installed, no dependencies yet), cargo tree (no dependencies).
- Known limits: verify.ps1 relies on LASTEXITCODE of cargo; privacy grep is a coarse text match.
