# Musigraph change log (log-first review, see project instructions)

## 2026-11-25 P0.1 workspace skeleton
- Branch feature/poc-audio-fullscreen. Files: Cargo.toml (workspace, edition 2024, rust-version 1.87, license Apache-2.0, forbid-style lints), rust-toolchain.toml (pins 1.100.0 + clippy/rustfmt), crates/{mg-dsp,mg-audio,mg-gfx,mg-engine} (name-constant smoke tests, engine prints a skeleton line), scripts/verify.ps1 (fmt, clippy -D warnings, test, build, Cargo.lock source check, no-eCapture grep gate, tracked-file allow-list, gitleaks).
- Build output goes to F:\musigraph\builds\target through a machine-local F:\musigraph\.cargo\config.toml OUTSIDE the repo (not committed, so no machine paths are published).
- No third-party dependencies yet (batch 1 is resolved in the next slice). mg-dsp has #![forbid(unsafe_code)]; workspace denies unsafe_op_in_unsafe_fn.
- Decisions: no scene trait, no shared types crate yet (plan v2). Toolchain pinned to the exact version so builds are reproducible; this installs a second 1.100.0 toolchain directory on F:.
- Verification run: scripts\verify.ps1 passes (fmt, clippy, 3 smoke tests, build, lockfile check, privacy gate, allow-list, gitleaks no leaks). Not run: osv-scanner (not installed, no dependencies yet), cargo tree (no dependencies).
- Known limits: verify.ps1 relies on LASTEXITCODE of cargo; privacy grep is a coarse text match.

## 2026-11-25 P0.1 review fixes (Sonnet High review: CHANGES REQUIRED -> addressed)
- verify.ps1: no committed machine path (gitleaks/osv resolved from PATH or $env:MUSIGRAPH_TOOLS, default ..\tools); the gate now FAILS if gitleaks or osv-scanner is missing (explicit -SkipSecretScan/-SkipOsv opt-outs); `--locked` on clippy/test/build/run; anchored lockfile source check (exact crates.io line); manifests scanned for [patch]/[replace]/git/registry/non-workspace path deps; anchored allow-list with extension filter under crates/scripts/docs/assets; broader capture-endpoint grep (eCapture|eAll|AUDIO_CAPTURE|DataFlow(1)); engine skeleton is executed; gitleaks history scan added; osv-scanner on Cargo.lock added.
- Cargo.toml: resolver "3" (MSRV-aware). rust-version 1.87 stays as the declared MSRV (wgpu's); the pinned toolchain is 1.100.0, so 1.87 is informational until a CI check exists.
- Deferred (documented): workspace-wide `unsafe_code = "deny"` with per-crate override lands with the first unsafe code (P1); tautological NAME smoke tests are scaffolding, replaced by real tests in P1/P2; `cargo tree -d` + build.rs/proc-macro allow-list diff are generated at dependency adoption.
- Reviewer claim not accepted: "gitleaks path does not exist on this machine" was wrong (it was installed to F:\musigraph\tools\gitleaks); the committed-absolute-path concern was valid and fixed.
- Verification: scripts\verify.ps1 passes end to end (fmt, clippy, 3 tests, build, run, source checks, privacy gate, allow-list, gitleaks x2, osv-scanner 2.7.0: 4 packages, no issues).

## 2026-11-25 P3.1 adapter report (first dependency: wgpu =30.0.2, features std,parking_lot,dx12,vulkan,gles,wgsl)
- Changes: crates/mg-gfx/src/report.rs (`adapter_report()` enumerates adapters separately for DX12/Vulkan/GL via wgpu and prints name/vendor/device/type/backend/driver/limits/features; small std `block_on` replaces a pollster dependency; 1 unit test), lib.rs exposes `report`, mg-engine main.rs accepts `--gpu-report` (unknown option exits 2), Cargo.toml/Cargo.lock (114 packages locked MSRV-aware by cargo for rust-version 1.87).
- Measured on the host (debug build, no device created yet): DX12 lists "NVIDIA GeForce GT 730" (DiscreteGpu, vendor 0x10de, device 0x1287, driver 30.0.14.7514) AND "Microsoft Basic Render Driver" (WARP, type Cpu); Vulkan lists the GT 730 (driver 475.14); GL lists "NVIDIA GeForce GT 730/PCIe/SSE2" (OpenGL 4.6.0 NVIDIA 475.14). Limits: max_texture_2d 16384, max_bind_groups 8, max_uniform_buffer 65536.
- Meaning: enumeration only. Device creation, surface, shader compilation and rendering are NOT yet verified (P3.2/P3.3). The CPU adapter (WARP) is listed by DX12: selection logic must reject DeviceType::Cpu so software rendering is never reported as hardware compatibility.
- Not run: release build, osv on the app lockfile with wgpu (scratch tree scanned clean), build.rs allow-list.
- verify.ps1: privacy grep made case-sensitive with word boundary on eAll (it falsely matched the word "really" in a comment; the gate itself worked).
