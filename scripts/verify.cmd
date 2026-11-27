@echo off
rem Reproducible launcher for the verification gate. The execution-policy bypass applies to THIS process
rem only (nothing in Windows is changed). Usage: scripts\verify.cmd [-SkipSecretScan] [-SkipOsv]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify.ps1" %*
exit /b %ERRORLEVEL%
