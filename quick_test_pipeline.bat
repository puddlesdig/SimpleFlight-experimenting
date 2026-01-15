@echo off
REM Quick pipeline test: Train -> Export -> Deploy
REM This validates the workflow with minimal training (just to test, not for performance)

echo ============================================================
echo Quick Pipeline Test
echo ============================================================
echo This will:
echo 1. Train a policy for 200 iterations (~5-10 minutes)
echo 2. Export it to deployment format
echo 3. Provide deployment command
echo.
echo NOTE: The policy won't fly well - this just tests the pipeline!
echo ============================================================
echo.

set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" exit /b

echo.
echo [1/3] Training policy (200 iterations, ~5-10 minutes)...
echo ============================================================
IsaacLab\isaaclab.bat -p IsaacLab\scripts\reinforcement_learning\train_simpleflight.py --task SimpleFlight-Track-Direct-v0 --num_envs 2048 --max_iterations 200

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Finding latest checkpoint...
echo ============================================================

REM Find the most recent model directory
for /f "delims=" %%i in ('dir /b /ad /o-d "logs\rsl_rl\simpleflight_track"') do (
    set LATEST_DIR=%%i
    goto :found
)

:found
if not defined LATEST_DIR (
    echo ERROR: No training logs found!
    pause
    exit /b 1
)

set CHECKPOINT=logs\rsl_rl\simpleflight_track\%LATEST_DIR%\model_200.pt
echo Found checkpoint: %CHECKPOINT%

if not exist "%CHECKPOINT%" (
    echo ERROR: Checkpoint file not found!
    echo Looking for: %CHECKPOINT%
    pause
    exit /b 1
)

echo.
echo [3/3] Exporting policy to deployment format...
echo ============================================================
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py --checkpoint "%CHECKPOINT%" --output models\quick_test_policy.pt

if errorlevel 1 (
    echo ERROR: Export failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Pipeline test complete.
echo ============================================================
echo.
echo Exported policy: models\quick_test_policy.pt
echo.
echo ============================================================
echo DEPLOYMENT TEST COMMANDS
echo ============================================================
echo.
echo 1. TEST MOTORS (without policy - verify hardware):
echo    cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_motors.py
echo.
echo 2. DEPLOY POLICY (3 seconds, hand-hold drone):
echo    cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor models\quick_test_policy.pt --duration 3 --thrust-offset 0.4
echo.
echo 3. ADJUST if motors don't spin (increase offset):
echo    cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor models\quick_test_policy.pt --duration 3 --thrust-offset 0.5
echo.
echo ============================================================
echo IMPORTANT: This policy is UNDERTRAINED (200 iterations only)
echo It will send commands, but won't fly well.
echo This test only verifies the pipeline works!
echo ============================================================
echo.
pause
