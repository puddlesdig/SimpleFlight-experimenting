# 🛑 Emergency Stop Guide for Crazyflie Deployment

## Multiple Layers of Safety Protection

### 1. **Software Emergency Stop (PRIMARY)** ⌨️

**Press `Ctrl+C` in the terminal**
- Immediately stops all motors
- Works from anywhere during flight
- Fastest software method (~50-100ms response)
- Script sends `send_stop_setpoint()` twice for reliability

```bash
# When running deployment, just press:
Ctrl + C
```

---

### 2. **Physical Emergency Stops** 🖐️

#### Option A: Hand Catch (For Low Altitude)
- If drone is <0.5m high and stable
- Cup your hands around it (avoid propellers!)
- Grab from underneath
- **RISK**: Spinning propellers can cut fingers
- **WHEN**: Drone is hovering low and you can safely reach it

#### Option B: Battery Disconnect
- Pull battery connector from Crazyflie
- **Instant power cut** - motors stop immediately
- **DOWNSIDE**: Drone will fall (crashes if flying)
- **WHEN**: Drone is on ground but spinning motors uncontrollably

#### Option C: Power Button Hold
- Hold power button for 3 seconds
- Less immediate than battery pull
- **WHEN**: Drone is on ground

---

### 3. **Crazyflie Client Emergency Stop** 🖥️

If you have **Crazyflie Client** open:
1. Keep client connected to drone
2. Click "Emergency Stop" button in GUI
3. Sends stop command via radio

**Setup**: Run client in parallel to deployment script:
```bash
# Terminal 1: Crazyflie Client (keep open)
# Terminal 2: Your deployment script
cf_deploy_env\Scripts\Activate.ps1
python IsaacLab\scripts\deployment\deploy_to_crazyflie.py ...
```

---

### 4. **Pre-Flight Safety Checklist** ✅

Before EVERY flight:

**Environment:**
- [ ] Clear 3m radius around drone
- [ ] No people nearby
- [ ] Soft landing surface (foam mat/grass)
- [ ] Safety net/cage if available
- [ ] Remove ceiling fans, hanging items

**Hardware:**
- [ ] Battery voltage >3.7V (check with client)
- [ ] All propellers secure and undamaged
- [ ] Correct propeller direction (CW/CCW)
- [ ] Motors spin freely (no grinding)
- [ ] Body intact, no cracks

**Software:**
- [ ] Crazyradio within 5m of drone
- [ ] Terminal window FOCUSED (for Ctrl+C)
- [ ] No other programs blocking USB
- [ ] Flight duration set SHORT (5-10s max initially)

**Your Readiness:**
- [ ] You know where Ctrl+C is
- [ ] Hand ready near keyboard
- [ ] Eyes on drone at all times
- [ ] Someone else present (optional but recommended)

---

### 5. **Safe Testing Progression** 📈

**Don't jump straight to flight!** Follow this sequence:

#### Step 1: Ground Test (Motors OFF)
```bash
# Test connection only - NO MOTOR COMMANDS
python IsaacLab\scripts\deployment\test_connection.py --uri radio://0/80/2M/E7E7E7E7E7
```
- Verifies radio link
- Checks battery
- Reads sensor data
- **Motors do NOT spin**

#### Step 2: Low Thrust Test (Won't Take Off)
```bash
# Modify deploy script to use thrust=15000 (below takeoff)
# Drone will sit on ground with motors spinning slowly
```
- Motors spin but drone stays grounded
- Verify policy is running
- Practice Ctrl+C stop
- **Duration: 3 seconds**

#### Step 3: Tethered Flight (HIGHLY RECOMMENDED)
- Attach fishing line/string to drone
- Hold the other end
- Limits maximum altitude to line length
- **Duration: 5 seconds**

#### Step 4: Free Flight (Short Duration)
```bash
python IsaacLab\scripts\deployment\deploy_to_crazyflie.py \
    --checkpoint cf_deploy_env\model_999.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --trajectory hover \
    --duration 5    # 5 seconds only!
```

#### Step 5: Longer Flights (If Policy is Good)
- Only after policy demonstrates stable hover
- Gradually increase duration: 10s → 20s → 30s

---

### 6. **What to Expect with Current Policy** ⚠️

Your policy trained for only 1000 iterations. **It will likely:**
- ❌ Flip immediately
- ❌ Fly in random directions
- ❌ Crash within 1 second
- ❌ Not hover at all

**Episode length was 1 step** = drone crashes immediately in simulation

**Recommendation**: 
1. Train to 5000+ iterations first
2. OR just test connection/low-thrust for now
3. Save real flight for when policy is ready

---

### 7. **Emergency Response Plan** 🚨

**If drone goes out of control:**

1. **IMMEDIATELY**: Press `Ctrl+C`
2. If Ctrl+C doesn't work (terminal not focused):
   - Click terminal window
   - Press Ctrl+C again
3. If still flying:
   - Pull battery connector (if accessible)
   - Use emergency stop in Crazyflie Client
4. If drone crashes:
   - Disconnect battery
   - Check for damage before next flight

**Post-Crash Checklist:**
- [ ] Inspect all propellers (replace if chipped)
- [ ] Check motor mounting (tighten screws)
- [ ] Test motor spin by hand (should be smooth)
- [ ] Check battery voltage (crash draws lots of current)
- [ ] Inspect body for cracks

---

### 8. **Legal/Ethical Considerations** ⚖️

- You are responsible for any damage/injury
- Test in private space with permission
- Don't fly near people, animals, or property
- Follow local drone regulations
- Document failures for learning (video recommended)

---

### 9. **Quick Command Reference** 📋

**Test connection (safe, no motors):**
```bash
cf_deploy_env\Scripts\Activate.ps1
python IsaacLab\scripts\deployment\test_connection.py
```

**Deploy policy (⚠️ USE WITH CAUTION):**
```bash
cf_deploy_env\Scripts\Activate.ps1
python IsaacLab\scripts\deployment\deploy_to_crazyflie.py \
    --checkpoint cf_deploy_env\model_999.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --trajectory hover \
    --duration 5
```

**Emergency stop during flight:**
```
Ctrl + C    (in the terminal running the script)
```

---

### 10. **When NOT to Fly** 🚫

Do NOT attempt flight if:
- Battery <3.7V
- Any propeller is damaged
- You haven't tested connection first
- You're alone (no one to help in emergency)
- Terminal is not in focus (can't Ctrl+C)
- You haven't read this safety guide
- Policy hasn't trained enough (current status)
- You don't have a clear emergency plan

---

## Summary

**Before you fly the current policy:**

1. ✅ The deployment environment is ready
2. ✅ Emergency stop (Ctrl+C) is implemented
3. ⚠️ **Policy is NOT flight-ready** (only 1000 iterations)

**Next steps:**
1. **Option A**: Test connection only (safe)
2. **Option B**: Continue training to 5000+ iterations
3. **Option C**: Attempt flight with EXTREME caution (expect crash)

**My recommendation**: Do Option A (test connection), then Option B (train more).
