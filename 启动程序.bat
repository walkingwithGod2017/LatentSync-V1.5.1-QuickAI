@echo off
rem 设置代理，7897可以替换成自己的端口地址
rem set http_proxy=http://127.0.0.1:7897 & set https_proxy=http://127.0.0.1:7897
rem set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%CD%\checkpoints"
set "TORCH_HOME=%CD%\checkpoints"
set "PYANNOTE_CACHE=%CD%\checkpoints\pyannote"
set "KERAS_BACKEND=torch"
set "SOX_PATH=%~dp0walkingwithai\sox"
set "FFMPEG_PATH=%~dp0walkingwithai\ffmpeg\bin"
set "PATH=%SOX_PATH%;%FFMPEG_PATH%;%PATH%"

rem 注意：在启动命令末尾添加了 --port %COM_PORT% 参数
.\walkingwithai\python.exe -s gradio_app.py

pause