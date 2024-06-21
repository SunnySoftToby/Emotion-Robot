@echo off
setlocal enabledelayedexpansion
for /r "影片資料夾路徑" %f in (*.mp4) do (
    for %%I in ("%~pf.") do (
        ffmpeg -i "%f" -vf fps=30   "output\%~nxI\%~nxI_%~nf_%05d.jpg" 
    )
)

