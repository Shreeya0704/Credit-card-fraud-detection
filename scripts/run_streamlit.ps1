try { (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique | % { Stop-Process -Id $_ -Force } } catch {}
$root = (Get-Location).Path
if (Test-Path ".\.venv\Scripts\python.exe") { $py = ".\.venv\Scripts\python.exe" } else { $py = "python" }
$env:PYTHONPATH = $root
$cmd = "& $py -m streamlit run src\dashboard\app.py --server.headless true"
Start-Process powershell -ArgumentList '-NoExit','-NoLogo','-Command', $cmd | Out-Null
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"
"STEP 5 OK"