param(
  [int]$Port = 8000,
  [string]$Csv = "data\raw\creditcard.csv",
  [switch]$KeepServer
)
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$env:PYTHONPATH = $root

# free port
try {
  $pids = (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
  if ($pids) { $pids | ForEach-Object { Stop-Process -Id $_ -Force } }
} catch { }

# start uvicorn as background job
$cmd = ".\.venv\Scripts\python.exe -m uvicorn src.service.app:app --host 0.0.0.0 --port $Port --log-level info"
$srv = Start-Job -Name uvicorn -ScriptBlock {
  param($c,$r)
  Set-Location $r
  & cmd.exe /c $c
} -ArgumentList $cmd,$root

# wait for health (parallel probe)
$healthy = $false
for ($i=0; $i -lt 60; $i++) {
  try {
    $res = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 2
    if ($res.status -eq 'ok') { $healthy = $true; break }
  } catch { Start-Sleep 1 }
}
if (-not $healthy) {
  Receive-Job $srv -Keep | Out-Host
  Stop-Job $srv -Force; Remove-Job $srv
  throw "API not healthy after 60s"
}

# score a single txn
$row = Import-Csv $Csv | Select-Object -First 1
if ($row.PSObject.Properties.Match('Class')) { $row.PSObject.Properties.Remove('Class') }
$body = @{ transactions = @($row) } | ConvertTo-Json -Depth 6
$result = Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/score" -ContentType 'application/json' -Body $body
$result | ConvertTo-Json

if (-not $KeepServer) { Stop-Job $srv -Force; Remove-Job $srv }
