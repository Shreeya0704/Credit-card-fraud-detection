$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py -m pip install --upgrade pip
& $py -m pip install pyarrow

$pyCode = @"
import pyarrow as pa
print("pyarrow", pa.__version__)
"@

$pyCode | Out-File -FilePath "scripts\temp_pyarrow_check.py" -Encoding utf8
& $py scripts\temp_pyarrow_check.py
Remove-Item "scripts\temp_pyarrow_check.py"

"STEP 4 OK"