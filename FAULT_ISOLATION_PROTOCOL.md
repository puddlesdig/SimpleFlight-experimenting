# Fault Isolation Protocol: Policy → Motor Actuation

## Problem Statement
**Symptom**: Motors do not move AT ALL when exported policy runs in deploy_standalone.py  
**Known Good**: test_motors.py successfully spins motors at thrust 40,000  
**Hypothesis**: Control pipeline / arming / command path failure (NOT thrust tuning)

---

## Isolation Stages (In Order)

### LAYER 1: Policy Output Sanity
**What to check**: Confirm inference runs, outputs are not NaN/zero, within expected ranges

**Where to log**: `policy_inference()` method in deploy_standalone.py

**Expected values**:
- Raw output: Varies by observation (unbounded)
- Tanh output: All values in [-1, 1]
- No NaN, no Inf
- Not all zeros (indicates dead network)

**Conclusion**:
- ✅ PASS: Policy outputs valid actions → proceed to LAYER 2
- ❌ FAIL (NaN/Inf): Model load corruption → check architecture match
- ❌ FAIL (all zeros): Dead network → verify checkpoint loading

**Test command**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor models\test_policy.pt --duration 3 --thrust-offset 0.5 --debug
```

---

### LAYER 2: Action Normalization
**What to check**: Confirm policy outputs are actually in [-1,1] after tanh

**Where to log**: `send_ctbr_command()` start, before mapping

**Expected values**:
- `action` array: All elements in [-1.0, 1.0]
- Specifically `action[3]` (thrust): Around -0.75 based on prior logs

**Conclusion**:
- ✅ PASS: Actions normalized correctly → proceed to LAYER 3
- ❌ FAIL (outside range): Tanh not applied or wrong → fix `torch.tanh()` call

---

### LAYER 3: Command Mapping
**What to check**: Policy outputs mapped to Crazyflie setpoint (RPYT) with correct units/signs

**Where to log**: `send_ctbr_command()` after thrust calculation, before send

**Expected values**:
- Roll/pitch/yaw rates: [-30, 30] deg/s (action * 30.0)
- Thrust: 
  - With offset 0.5: `((-0.75+1)/2 + 0.5) * 65535 = 40,959`
  - Should be > 20,000 (motor threshold)
  - Should be > 0 (not clipped to zero)

**Conclusion**:
- ✅ PASS: Thrust > 20,000 and valid RPYT → proceed to LAYER 4
- ❌ FAIL (thrust = 0): Offset/scale math wrong → check calculation order
- ❌ FAIL (thrust < 20,000): Offset too low → increase to 0.6+

---

### LAYER 4: Radio Transport
**What to check**: Packets sent over Crazyradio, received by CF, acknowledged

**Where to log**: `send_ctbr_command()` inside try-except around `send_setpoint()`

**Expected values**:
- `send_setpoint()` executes without exception
- cflib debug logs (if enabled) show packet ACKs
- No "No ACK" or "Too many packets lost" warnings

**Conclusion**:
- ✅ PASS: No exceptions, packets flowing → proceed to LAYER 5
- ❌ FAIL (exceptions): Radio link broken → check connection, reduce rate

**Bypass test**: Use `test_policy_bypass.py` to send hardcoded setpoint
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_policy_bypass.py --duration 3
```
Expected: Motors spin (proves radio + commander work, isolates policy)

---

### LAYER 5: Commander Path
**What to check**: Setpoints reach correct commander API, not rejected

**Where to log**: After `commander.enHighLevel` set, read back param

**Expected values**:
- `commander.enHighLevel = 0` (confirmed by readback)
- No cflib errors about "Invalid setpoint" or "Command rejected"

**Conclusion**:
- ✅ PASS: enHighLevel disabled → proceed to LAYER 6
- ❌ FAIL (enHighLevel = 1): Param not set → check cflib version, param name
- ❌ FAIL (rejection): Wrong API → verify using `send_setpoint()` not `send_hover()`

---

### LAYER 6: Motor Enable / Arming
**What to check**: Firmware allows motors to spin (estimator ready, no safety lock)

**Where to log**: After connecting, before control loop

**Expected values**:
- `supervisor.canFly = 1` (drone ready)
- `supervisor.isFlying = 0` initially (on ground)
- Battery voltage > 3.0V per cell
- Estimator state: RUNNING (not UNINITIALIZED)

**Conclusion**:
- ✅ PASS: canFly=1, estimator ready → proceed to LAYER 7
- ❌ FAIL (canFly=0): Safety engaged → check Flow Deck, battery, startup logs
- ❌ FAIL (estimator not ready): Wait longer (>2s) or check sensor health

**Firmware check**: 
- Crazyflie firmware has NO separate "arming" command for low-level RPYT
- Motors spin if `thrust > ~10,000` and `commander.enHighLevel = 0`
- No takeoff mode gating for `send_setpoint()`

---

### LAYER 7: Timing / Watchdog
**What to check**: Setpoint rate high enough, watchdog not zeroing outputs

**Where to log**: In control loop, measure actual loop time

**Expected values**:
- Loop time: < 10ms (target: 10ms for 100Hz)
- No gaps > 500ms (watchdog timeout)
- Continuous send without interruption

**Conclusion**:
- ✅ PASS: Loop time < 10ms, continuous → motors SHOULD spin
- ❌ FAIL (loop > 10ms): Processing too slow → reduce inference overhead
- ❌ FAIL (intermittent send): Blocking I/O → check logging, sensors

**Watchdog behavior**:
- Crazyflie firmware zeros setpoint if no command for ~500ms
- At 100Hz (10ms), watchdog should never trigger
- If motors spin briefly then stop: watchdog timing issue

---

## Deliverables

### A) Failure Points Table

| Layer | Symptom | Likely Cause | Check | Fix |
|-------|---------|--------------|-------|-----|
| 1 | NaN/Inf actions | Model load failure | Log raw output | Verify architecture matches checkpoint |
| 1 | All-zero actions | Dead network | Log tanh output | Check `load_weights()` mapping |
| 2 | Actions > 1 | No tanh | Log action range | Apply `torch.tanh()` |
| 3 | Thrust = 0 | Offset wrong | Log thrust calc | Fix order: `(norm + offset) * scale` |
| 3 | Thrust < 20k | Offset too low | Log final thrust | Increase offset to 0.6+ |
| 4 | send_setpoint() exception | Radio failure | Log exception | Check connection, reduce rate |
| 5 | enHighLevel = 1 | Param not set | Read back param | Verify cflib version, param name |
| 6 | canFly = 0 | Safety engaged | Read supervisor.* | Check sensors, battery, wait longer |
| 7 | Loop > 10ms | Processing slow | Measure loop time | Reduce logging, optimize inference |

### B) Known-Good Test (Already Working)

**File**: `test_motors.py`  
**Result**: ✅ Motors spin at thrust 40,000

**Command**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_motors.py
```

**Expected logs**:
```
Connected to radio://0/80/2M/E7E7E7E7E7
Disabling high-level commander...
Sending thrust: 40000
Motors should be spinning!
```

**What this proves**:
- Radio link works
- Commander API works (`send_setpoint`)
- Motors physically work
- `commander.enHighLevel = 0` works
- No arming/safety blocking

### C) Policy-Free Test (Isolate Mapping)

**File**: `test_policy_bypass.py` (created)

**Command**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_policy_bypass.py --duration 3
```

**What it does**:
- Uses EXACT same command path as deploy_standalone
- Bypasses policy inference
- Sends hardcoded RPYT: `[0, 0, 0, 40959]` (matching expected policy output with offset 0.5)

**Expected result**:
- Motors spin (proves command mapping + transport work)

**If this WORKS but deploy_standalone DOESN'T**:
- Problem is in LAYER 1 (policy inference) or LAYER 2 (normalization)
- Check: Model loading, tanh application, NaN outputs

**If this ALSO FAILS**:
- Problem is in LAYER 4+ (radio/commander/arming)
- Compare with test_motors.py to find difference

### D) Instrumented Deploy Script

**File**: `deploy_standalone.py` (updated with --debug flag)

**Command**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor models\test_policy.pt --duration 3 --thrust-offset 0.5 --debug
```

**Debug output shows**:
- [LAYER 1] Raw policy output, tanh output, NaN check
- [LAYER 2] Action range validation
- [LAYER 3] RPYT mapping, thrust calculation steps
- [LAYER 4] send_setpoint() success/exception
- [LAYER 5] commander.enHighLevel readback
- [LAYER 6] supervisor.canFly, supervisor.isFlying
- [LAYER 7] Loop timing measurements

**Analysis workflow**:
1. Run with `--debug` flag
2. Scan logs for `[LAYER X]` markers
3. Find first FAILURE or WARNING
4. Apply fix from table above
5. Repeat until motors spin

---

## Execution Plan

### Step 1: Run Policy-Free Test
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\test_policy_bypass.py --duration 3
```

**Expected**: Motors spin (proves LAYER 4-7 work)  
**If fails**: Compare with test_motors.py, check radio/commander

### Step 2: Run Instrumented Policy Test
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor models\test_policy.pt --duration 3 --thrust-offset 0.5 --debug
```

**Expected**: Detailed logs showing layer-by-layer validation  
**Action**: Find first FAILURE/WARNING, apply fix

### Step 3: Analyze Logs
**Look for**:
- `[LAYER 1] FAILURE: NaN/Inf` → Model loading issue
- `[LAYER 2] FAILURE: Action outside [-1,1]` → Tanh not applied
- `[LAYER 3] WARNING: Thrust mapped to ZERO` → Offset calculation wrong
- `[LAYER 3] WARNING: Thrust XXX below motor threshold` → Offset too low
- `[LAYER 5] FAILURE: High-level commander still enabled` → Param not set
- `[LAYER 6] WARNING: supervisor.canFly is not 1` → Safety/estimator issue
- `[LAYER 7] Loop time XXXms exceeds target` → Processing too slow

### Step 4: Apply Fix
**Based on first failure**:
- LAYER 1/2: Check model loading, verify tanh
- LAYER 3: Increase offset to 0.6, verify calculation
- LAYER 4: Check radio connection, reduce rate
- LAYER 5: Verify cflib version, param name
- LAYER 6: Wait longer (>5s), check sensors
- LAYER 7: Reduce logging, optimize inference

---

## Likely Culprits (Ranked)

### 1. LAYER 3: Thrust Calculation Bug (High Probability)
**Why**: Motors don't move suggests thrust = 0 or < threshold  
**Evidence**: Prior tests showed action[3] ≈ -0.75  
**Check**: Log final thrust value in `send_ctbr_command()`  
**Expected**: ~40,959 with offset 0.5  
**Fix**: If thrust < 20,000, increase offset OR fix calculation order

### 2. LAYER 1: Policy Output Always Same (Medium Probability)
**Why**: Undertrained policy (1000 iter) might output constant actions  
**Evidence**: No prior logs showing action variation  
**Check**: Log action over time, verify it changes  
**Expected**: Actions vary based on observation  
**Fix**: If constant, policy is too simple → train longer

### 3. LAYER 5: Commander State (Medium Probability)
**Why**: `commander.enHighLevel` might not disable properly  
**Evidence**: No confirmation it was read back  
**Check**: Read param after setting  
**Expected**: Returns '0'  
**Fix**: If still '1', cflib version issue or wrong param name

### 4. LAYER 6: Supervisor Lock (Low Probability)
**Why**: test_motors.py works, so likely not safety  
**Evidence**: Same radio/commander path  
**Check**: Read `supervisor.canFly`  
**Expected**: '1'  
**Fix**: If '0', wait longer or check Flow Deck

### 5. LAYER 4: Radio Transport (Very Low Probability)
**Why**: test_motors.py proves radio works  
**Evidence**: Same cflib version, same URI  
**Check**: Compare packet rate with test_motors.py  
**Fix**: Reduce loop rate if packet loss

---

## Success Criteria

**Test passes when**:
1. `test_policy_bypass.py` → Motors spin (proves LAYER 4-7)
2. `deploy_standalone.py --debug` → Motors spin (proves LAYER 1-3)
3. Debug logs show:
   - [LAYER 1] ✅ Valid action outputs
   - [LAYER 2] ✅ Actions in [-1, 1]
   - [LAYER 3] ✅ Thrust > 40,000
   - [LAYER 4] ✅ send_setpoint() no exceptions
   - [LAYER 5] ✅ enHighLevel = 0
   - [LAYER 6] ✅ canFly = 1
   - [LAYER 7] ✅ Loop time < 10ms

**Root cause identified when**:
- One layer consistently shows FAILURE/WARNING
- Applying fix from table resolves issue
- Motors spin after fix

---

## Next Steps After Motors Spin

1. **Validate hover**: Increase duration to 10s, check stability
2. **Tune thrust**: Fine-tune offset based on actual hover height
3. **Train properly**: Fix IsaacLab deps, train 5000 iterations
4. **Physics calibration**: Update sim to match 40,000 hover thrust
5. **Domain randomization**: Add mass/thrust variation

**DO NOT** proceed to training/calibration until motors physically spin with current policy.
