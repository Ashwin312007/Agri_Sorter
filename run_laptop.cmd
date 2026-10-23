@echo off
cd /d "%~dp0"
echo ==============================================
echo Apple Fruit Damage Detector (Live Webcam)
echo Press 'q' in the camera window to quit.
echo ==============================================
.\.venv\Scripts\python.exe -u detect.py --model best.onnx --cam 0 --conf 0.25
pause
