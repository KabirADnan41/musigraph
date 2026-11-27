# Live-audio smoke test (P1/P2): plays a LOW-VOLUME synthetic tone through the default output while
# `mg-engine --print-levels` captures it via WASAPI loopback. Test fixture only (labelled synthetic);
# nothing is recorded: the engine prints derived numbers only. Never run as administrator.
param(
  [double]$Freq = 440,
  [double]$Amp = 0.05,       # about -26 dBFS: quiet
  [int]$ToneSeconds = 3,
  [int]$Seconds = 9,
  [string]$Exe = ''
)
$ErrorActionPreference = 'Stop'
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
if ($Amp -gt 0.2 -or $Amp -le 0) { throw 'Amp must be in (0, 0.2] (low-volume fixture only)' }
$root = Split-Path $PSScriptRoot -Parent
if (-not $Exe) { $Exe = Join-Path (Split-Path $root -Parent) 'builds\target\release\mg-engine.exe' }
if (-not (Test-Path $Exe)) { throw "engine not found: $Exe" }
$tmp = Join-Path (Split-Path $root -Parent) 'temp'
New-Item -ItemType Directory -Force $tmp | Out-Null
$wav = Join-Path $tmp 'tone-test.wav'
$rate = 48000; $n = $rate * $ToneSeconds
$ms = New-Object IO.MemoryStream; $bw = New-Object IO.BinaryWriter $ms
$bw.Write([Text.Encoding]::ASCII.GetBytes('RIFF')); $bw.Write([int](36 + 2 * $n)); $bw.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
$bw.Write([int]16); $bw.Write([int16]1); $bw.Write([int16]1); $bw.Write([int]$rate); $bw.Write([int]($rate * 2)); $bw.Write([int16]2); $bw.Write([int16]16)
$bw.Write([Text.Encoding]::ASCII.GetBytes('data')); $bw.Write([int](2 * $n))
$fade = [int]($rate * 0.02)
for ($i = 0; $i -lt $n; $i++) {
  $env = [Math]::Min(1.0, [Math]::Min($i, $n - 1 - $i) / $fade)
  $bw.Write([int16][Math]::Round(32767 * $Amp * $env * [Math]::Sin(2 * [Math]::PI * $Freq * $i / $rate)))
}
$bw.Flush(); [IO.File]::WriteAllBytes($wav, $ms.ToArray())
$out = Join-Path $tmp 'live-levels.out.txt'
$p = Start-Process -FilePath $Exe -ArgumentList @('--print-levels', '--seconds', $Seconds) -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError (Join-Path $tmp 'live-levels.err.txt')
try {
  Start-Sleep -Seconds 2
  $sp = New-Object Media.SoundPlayer $wav
  $sp.PlaySync()
  $null = $p.WaitForExit(($Seconds + 15) * 1000)
} finally {
  if (-not $p.HasExited) { try { $p.Kill() } catch { } }
  Remove-Item $wav -ErrorAction SilentlyContinue   # do not leave audio files behind
}
Get-Content $out
Get-Content (Join-Path $tmp 'live-levels.err.txt')
