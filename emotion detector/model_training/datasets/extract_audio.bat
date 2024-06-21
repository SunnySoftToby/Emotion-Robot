@echo off
setlocal enabledelayedexpansion


for /r "<影片資料夾位址>" %%f in (*.mp4) do (
    REM 獲得wav與資料夾編號
    for %%I in ("%%~pf.") do (
        set "output_folder=D:\CMU-MultimodalSDK\project\raw_dataset\Raw\audios\%%~nxI"
       
        echo output_folder

        if not exist "!output_folder!" (
            mkdir "!output_folder!"
        )

        rem only use left side
        "D:\ffmpeg-7.0-essentials_build\ffmpeg-7.0-essentials_build\bin\ffmpeg.exe" -i "%%f" -ar 16000 -filter_complex "channelsplit=channel_layout=stereo:channels=FL[left]" -map "[left]" -t 10 "!output_folder!\%%~nxI_%%~nf.wav"
        
        rem right side channel
        rem ffmpeg -i "%%f" -ar 16000 -filter_complex "channelsplit=channel_layout=stereo:channels=FR[right]" -map "[right]" "!output_folder!\%%~nxI_%%~nf_right.wav"
    )
)
