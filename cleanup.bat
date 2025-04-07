@echo off
setlocal

echo === CLEANING UP REDUNDANT FILES ===
echo.

REM List of files to remove
set "files_to_remove=activate_env.bat fix_dependencies.bat fix_huggingface.bat fix_token.bat fix_token_all.bat fix_token_format.bat run_tests.bat run_transcribe_live.bat set_token_direct.bat starta_server.bat stoppa_server.bat test_audio.bat test_voice.bat"

for %%f in (%files_to_remove%) do (
    if exist "%%f" (
        echo Removing: %%f
        del "%%f"
    )
)

echo.
echo Cleanup complete!
echo The project structure has been streamlined.
echo Please use setup.bat and run.bat instead.
pause
