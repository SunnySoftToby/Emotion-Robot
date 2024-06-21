@echo off
setlocal enabledelayedexpansion


for /r "<影片資料夾位址>" %%f in (*.mp4) do (
    REM 獲得wav與資料夾編號
    for %%I in ("%%~pf.") do (
        set "output_folder=<圖片輸出資料夾位址>\audios\%%~nxI"
       
        echo output_folder

        if not exist "!output_folder!" (
            mkdir "!output_folder!"
        )

        rem only use left side
        "ffmpeg -i "%%f" -ar 16000 -filter_complex "channelsplit=channel_layout=stereo:channels=FL[left]" -map "[left]" -t 10 "!output_folder!\%%~nxI_%%~nf.wav"
        
    )
)
