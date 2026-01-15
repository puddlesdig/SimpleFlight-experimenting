@echo off
REM Quick start guide for SimpleFlight calibration and deployment

echo ========================================
echo SimpleFlight Calibration Workflow
echo ========================================
echo.

:menu
echo What would you like to do?
echo.
echo 1. Measure hover thrust (Phase 1)
echo 2. Train new policy (Phase 3)
echo 3. Export trained policy (Phase 4)
echo 4. Deploy to drone (Phase 5)
echo 5. Test motors only
echo 6. Open calibration guide
echo 7. Exit
echo.

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" goto measure
if "%choice%"=="2" goto train
if "%choice%"=="3" goto export
if "%choice%"=="4" goto deploy
if "%choice%"=="5" goto test_motors
if "%choice%"=="6" goto guide
if "%choice%"=="7" goto end

echo Invalid choice!
goto menu

:measure
echo.
echo ========================================
echo Measuring Hover Thrust
echo ========================================
echo SAFETY: Hold drone firmly in your hand!
echo.
pause
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\measure_hover_thrust.py
goto menu

:train
echo.
echo ========================================
echo Training Policy
echo ========================================
echo This will take 45-90 minutes...
echo.
set /p iterations="Enter max iterations (default 5000): "
if "%iterations%"=="" set iterations=5000

D:\IsaacSim\python.bat IsaacLab\scripts\train_simpleflight.py --task SimpleFlight-Track-Direct-v0 --num_envs 2048 --max_iterations %iterations%
goto menu

:export
echo.
echo ========================================
echo Export Policy
echo ========================================
echo.
set /p checkpoint="Enter checkpoint path (e.g., logs/rsl_rl/.../model_5000.pt): "
set /p output="Enter output path (default: models/my_policy.pt): "
if "%output%"=="" set output=models\my_policy.pt

cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py --checkpoint %checkpoint% --output %output%
goto menu

:deploy
echo.
echo ========================================
echo Deploy to Drone
echo ========================================
echo.
set /p actor="Enter actor path (default: models/actor_standalone.pt): "
if "%actor%"=="" set actor=models\actor_standalone.pt

set /p duration="Enter duration in seconds (default: 3): "
if "%duration%"=="" set duration=3

set /p offset="Enter thrust offset (default: 0.4): "
if "%offset%"=="" set offset=0.4

set /p scale="Enter thrust scale (default: 1.0): "
if "%scale%"=="" set scale=1.0

echo.
echo SAFETY: Ensure clear 3x3m area!
echo Press Ctrl+C anytime to emergency stop.
echo.
pause

cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor %actor% --duration %duration% --thrust-offset %offset% --thrust-scale %scale%
goto menu

:test_motors
echo.
echo ========================================
echo Test Motors
echo ========================================
echo SAFETY: Remove propellers first!
echo.
pause
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_motors.py
goto menu

:guide
echo.
echo Opening calibration guide...
start IsaacLab\scripts\deployment\CALIBRATION_GUIDE.md
goto menu

:end
echo.
echo Goodbye!
exit /b
