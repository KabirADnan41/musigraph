# Musigraph

A native Windows music visualizer built with Rust, featuring immersive fullscreen audio-reactive scenes and, later, customizable desktop widgets.

**Status: early development (pre-alpha).** There is no usable release yet. The first milestone is a native fullscreen visualizer that reacts to system audio.

## Goals
- Fullscreen, widescreen, atmospheric scenes with original procedural graphics and several distinct visual styles.
- Native Rust engine: WASAPI system-audio capture, shared FFT analysis, GPU rendering through wgpu.
- Efficient on modest hardware.
- Desktop widgets and an on-demand dashboard are planned for later milestones.

## Platform
Windows 10/11, x86_64. Builds use the Rust MSVC toolchain.

## Development
Work happens on feature branches; `main` holds reviewed milestones. Contributions are not being accepted yet.

## Licence
Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 KabirADnan41. See [NOTICE](NOTICE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
