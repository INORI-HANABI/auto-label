@echo off
for /f "delims=" %%i in ("%cd%") do set folder=%%~ni
echo %folder%
md out-%folder%
for /R %%v IN (*.h264) do ( ffmpeg -i %%v -vcodec copy -f avi "out-%folder%\%%~nv.avi")
