@echo off
rem Double-click to open the demo at http://127.0.0.1:8765 (first launch sets up the environment and the data slice).
cd /d "%~dp0.."
if not exist .venv-demo\Scripts\python.exe (
  uv venv .venv-demo -p 3.12 || goto :err
  uv pip install -p .venv-demo\Scripts\python.exe -r requirements.txt || goto :err
)
if not exist local_data\demo\data\train_source1.parquet .venv-demo\Scripts\python demo\make_sample.py || goto :err
set PYTHONUTF8=1
.venv-demo\Scripts\python demo\server.py
goto :eof
:err
echo Setup failed. See the message above.
pause
