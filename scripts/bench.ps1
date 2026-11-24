# Runs one Musigraph GPU benchmark/test run and records process metrics plus an optional screenshot.
# Example: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bench.ps1 -Backend dx12 -Scale 1.0 -Vsync off -Seconds 10 -Shot
# Never run as administrator. Output: <OutDir>\results.csv (+ PNG screenshots). OutDir defaults to ..\benchmarks next to the repository.
param(
  [ValidateSet('auto', 'dx12', 'vulkan', 'gl')][string]$Backend = 'auto',
  [ValidateSet('bench', 'test')][string]$Mode = 'bench',
  [double]$Scale = 1.0,
  [ValidateSet('on', 'off')][string]$Vsync = 'on',
  [int]$Seconds = 10,
  [switch]$Windowed,
  [switch]$Shot,
  [switch]$Vram,
  [ValidatePattern('^[\w-]*$')][string]$Tag = '',
  [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture  # '.' decimals in args and CSV
# CopyFromScreen captures the whole primary screen, so screenshots are only allowed for the fullscreen run (nothing else visible).
if ($Shot -and $Windowed) { throw '-Shot is only allowed for fullscreen runs (a windowed capture would include other windows)' }
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutDir) { $OutDir = Join-Path (Split-Path $root -Parent) 'benchmarks' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$exe = Join-Path (Split-Path $root -Parent) 'builds\target\release\mg-engine.exe'
if (-not (Test-Path $exe)) { throw "release binary not found: $exe (cargo build --release -p mg-engine)" }
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if ((New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'refusing to run elevated' }

function Vram { try { [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()) } catch { -1 } }
$flag = if ($Mode -eq 'bench') { '--gpu-bench' } else { '--gpu-test' }
$argv = @($flag, '--backend', $Backend, '--vsync', $Vsync, '--seconds', $Seconds)
if ($Mode -eq 'bench') { $argv += @('--scale', ('{0:0.00}' -f $Scale)); if ($Windowed) { $argv += '--windowed' } } elseif (-not $Windowed) { $argv += '--fullscreen' }
$kind = if ($Shot) { 'shot' } elseif ($Vram) { 'vram' } else { 'meas' }
$name = "{0}-{1}-{2}-s{3:0.00}-v{4}{5}" -f $kind, $Mode, $Backend, $Scale, $Vsync, $(if ($Tag) { "-$Tag" } else { '' })
$out = Join-Path $OutDir "$name.out.txt"; $err = Join-Path $OutDir "$name.err.txt"
$vb = Vram
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $exe -ArgumentList $argv -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
$null = $p.Handle  # cache the handle so ExitCode and PeakWorkingSet64 stay readable after exit
$vd = -1; $png = ''
$peak = 0; $killed = $false; $deadline = (Get-Date).AddSeconds($Seconds + 30)
try {
if ($Shot) {
  Start-Sleep -Milliseconds 4500
  $vd = Vram
  Add-Type -AssemblyName System.Windows.Forms, System.Drawing
  Add-Type -Namespace Mg -Name Win -MemberDefinition '[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern System.IntPtr GetForegroundWindow();'
  $p.Refresh()
  # Capture only when the Musigraph window is the foreground window, so no other desktop content is recorded.
  if ($p.HasExited -or $p.MainWindowHandle -eq [IntPtr]::Zero -or [Mg.Win]::GetForegroundWindow() -ne $p.MainWindowHandle) {
    Write-Warning 'screenshot skipped: Musigraph window is not the foreground window'
    $Shot = $false
  }
}
if ($Shot) {
  $b = [Windows.Forms.Screen]::PrimaryScreen.Bounds
  $bmp = New-Object Drawing.Bitmap $b.Width, $b.Height
  $g = [Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($b.Location, [Drawing.Point]::Empty, $b.Size); $g.Dispose()
  $png = Join-Path $OutDir "$name.png"; $bmp.Save($png, [Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
} elseif ($Vram) { Start-Sleep -Milliseconds 4500; $vd = Vram }
# Screenshots and VRAM sampling (nvidia-smi spawn) disturb timing: use separate -Shot / -Vram runs, never measurement runs.
while (-not $p.HasExited) {
  try { $p.Refresh(); $peak = [Math]::Max($peak, [int]($p.PeakWorkingSet64 / 1MB)) } catch { }
  if ((Get-Date) -gt $deadline) { $p.Kill(); $killed = $true; break }  # emergency cleanup: hard timeout
  Start-Sleep -Milliseconds 500
}
} finally { try { if (-not $p.HasExited) { $p.Kill() } } catch { } }  # never leave a fullscreen child behind on script error / Ctrl-C
$sw.Stop()
$p.WaitForExit()
$cpu = $p.TotalProcessorTime.TotalSeconds
$txt = (Get-Content $out -Raw -ErrorAction SilentlyContinue) + (Get-Content $err -Raw -ErrorAction SilentlyContinue)
$bench = ($txt -split "`n" | Where-Object { $_ -like 'BENCH*' } | Select-Object -First 1)
$line = "{0},{1},{2},{3},{4},exit={5},killed={6},wall_s={7:N1},cpu_s={8:N2},cpu_pct_of_1core={9:N1},peak_ws_mb={10},vram_before_mb={11},vram_during_mb={12},{13}" -f (Get-Date -Format s), $Mode, $Backend, $Scale, $Vsync, $p.ExitCode, $killed, $sw.Elapsed.TotalSeconds, $cpu, (100 * $cpu / $sw.Elapsed.TotalSeconds), $peak, $vb, $vd, ([string]$bench -replace ',', ';')
Add-Content (Join-Path $OutDir 'results.csv') $line -Encoding ascii
$line
if ($png) { "screenshot: $png" }
Get-Process mg-engine -ErrorAction SilentlyContinue | ForEach-Object { 'LEFTOVER PROCESS ' + $_.Id }
