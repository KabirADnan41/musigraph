# Musigraph verification gate. Run: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1 [-SkipSecretScan] [-SkipOsv]
# Needs RUSTUP_HOME/CARGO_HOME/PATH from the user environment. Never run as administrator.
# Tool locations: PATH first, else $env:MUSIGRAPH_TOOLS (default: ..\tools next to this repository).
param([switch]$SkipSecretScan, [switch]$SkipOsv)
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$tools = if ($env:MUSIGRAPH_TOOLS) { $env:MUSIGRAPH_TOOLS } else { Join-Path (Split-Path $root -Parent) 'tools' }
function Step($name, [scriptblock]$b) { Write-Host "== $name"; & $b; if ($LASTEXITCODE -ne 0) { throw "FAILED: $name" } }
function Find-Tool($cmd, $rel) { $c = Get-Command $cmd -ErrorAction SilentlyContinue; if ($c) { return $c.Source }; $p = Join-Path $tools $rel; if (Test-Path $p) { return $p }; return $null }
Step 'fmt'    { cargo fmt --all -- --check }
Step 'clippy' { cargo clippy --workspace --all-targets --locked -- -D warnings }
Step 'test'   { cargo test --workspace --locked }
Step 'build'  { cargo build --workspace --locked }
Step 'run engine skeleton' { cargo run --locked -q -p mg-engine }
& {
$ErrorActionPreference = 'Stop'
Write-Host '== dependency sources (crates.io registry only; no git/path-outside/patch)'
$bad = Select-String -Path Cargo.lock -Pattern '^source = ' | Where-Object { $_.Line -ne 'source = "registry+https://github.com/rust-lang/crates.io-index"' }
if ($bad) { $bad | ForEach-Object { $_.Line }; throw 'FAILED: non-crates.io dependency source' }
$manifests = Get-ChildItem -Recurse -Filter Cargo.toml | Where-Object { $_.FullName -notmatch '[\\/]target[\\/]' }
if (-not $manifests) { throw 'FAILED: no Cargo.toml found for the manifest checks' }
$mbad = $manifests | Select-String -Pattern '^\s*\[patch|^\s*\[replace|\bgit\s*=|\bregistry\s*=' 
if ($mbad) { $mbad | ForEach-Object { $_.ToString() }; throw 'FAILED: git/patch/alternative-registry dependency in a manifest' }
$pbad = $manifests | Select-String -Pattern 'path\s*=\s*"' | Where-Object { $_.Line -notmatch 'path\s*=\s*"\.\./mg-[a-z]+"' }
if ($pbad) { $pbad | ForEach-Object { $_.ToString() }; throw 'FAILED: path dependency outside the workspace crates' }
Write-Host '== privacy gate: no capture (microphone) endpoint API'
$hit = Get-ChildItem crates -Recurse -Filter *.rs | Select-String -CaseSensitive -Pattern 'eCapture|\beAll\b|AUDIO_CAPTURE|DataFlow\s*\(\s*1|build_input_stream|default_input_device|Direction::Capture'
if ($hit) { $hit | ForEach-Object { $_.ToString() }; throw 'FAILED: capture endpoint referenced (render-endpoint loopback only)' }
Write-Host '== tracked-file allow-list'
$ok = '^(\.gitignore|LICENSE|NOTICE|README\.md|THIRD-PARTY-NOTICES\.md|log\.md|rust-toolchain\.toml|Cargo\.toml|Cargo\.lock)$|^(crates|scripts|docs|assets)/.+\.(rs|toml|md|ps1|wgsl|txt|json|csv)$'
$unexpected = git ls-files | Where-Object { $_ -notmatch $ok }
if ($unexpected) { $unexpected; throw 'FAILED: unexpected tracked files' }
}
if (-not $SkipSecretScan) {
  $gl = Find-Tool 'gitleaks' 'gitleaks\gitleaks.exe'
  if (-not $gl) { throw 'FAILED: gitleaks not found (PATH or MUSIGRAPH_TOOLS); use -SkipSecretScan only deliberately' }
  Step 'gitleaks (working files)' { & $gl dir . --redact --no-banner }
  Step 'gitleaks (history)'       { & $gl git . --redact --no-banner }
}
if (-not $SkipOsv) {
  $osv = Find-Tool 'osv-scanner' 'osv\osv.exe'
  if (-not $osv) { throw 'FAILED: osv-scanner not found (PATH or MUSIGRAPH_TOOLS); use -SkipOsv only deliberately' }
  Step 'osv-scanner (Cargo.lock)' { & $osv scan source --lockfile Cargo.lock }
}
Write-Host 'VERIFY OK'
