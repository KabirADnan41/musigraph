# Musigraph verification gate. Run from anywhere: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
# Needs RUSTUP_HOME/CARGO_HOME/PATH from the user environment. Never run as administrator.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Step($name, [scriptblock]$b) { Write-Host "== $name"; & $b; if ($LASTEXITCODE -ne 0) { throw "FAILED: $name" } }
Step 'fmt'    { cargo fmt --all -- --check }
Step 'clippy' { cargo clippy --workspace --all-targets -- -D warnings }
Step 'test'   { cargo test --workspace }
Step 'build'  { cargo build --workspace }
Write-Host '== lockfile sources (crates.io only)'
if (Test-Path Cargo.lock) {
  $bad = Select-String -Path Cargo.lock -Pattern '^source = ' | Where-Object { $_.Line -notmatch 'registry\+https://github.com/rust-lang/crates.io-index' }
  if ($bad) { $bad | ForEach-Object { $_.Line }; throw 'FAILED: non-crates.io dependency source' }
}
Write-Host '== privacy gate: no capture (microphone) endpoint API'
$hit = Get-ChildItem crates -Recurse -Filter *.rs | Select-String -Pattern 'eCapture'
if ($hit) { $hit | ForEach-Object { $_.ToString() }; throw 'FAILED: capture endpoint referenced (render-endpoint loopback only)' }
Write-Host '== tracked-file allow-list'
$unexpected = git ls-files | Where-Object { $_ -notmatch '^(\.gitignore|LICENSE|NOTICE|README\.md|THIRD-PARTY-NOTICES\.md|log\.md|rust-toolchain\.toml|Cargo\.(toml|lock)|crates/|scripts/|docs/|assets/)' }
if ($unexpected) { $unexpected; throw 'FAILED: unexpected tracked files' }
$gl = 'F:\musigraph\tools\gitleaks\gitleaks.exe'
if (Test-Path $gl) { Step 'gitleaks' { & $gl dir . --redact --no-banner } } else { Write-Warning 'gitleaks not found; skipped' }
Write-Host 'VERIFY OK'
