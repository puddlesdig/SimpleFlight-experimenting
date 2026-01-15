#!/usr/bin/env python3
"""Measure the actual hover thrust of your Crazyflie.

This script incrementally increases thrust to find the minimum value
needed for the drone to just barely lift off. This calibrates sim-to-real
transfer by measuring real-world hover thrust.

SAFETY: Hold the drone or use a test stand. Motors will spin!
"""

import argparse
import time
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_hover_thrust(uri):
    """Find hover thrust by gradually increasing thrust."""
    
    cflib.crtp.init_drivers()
    
    logger.info(f"Connecting to {uri}...")
    
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        
        # Disable high-level commander
        cf.param.set_value('commander.enHighLevel', '0')
        time.sleep(0.1)
        
        logger.info("\n" + "="*60)
        logger.info("HOVER THRUST CALIBRATION")
        logger.info("="*60)
        logger.info("\nInstructions:")
        logger.info("1. Hold the drone firmly in your hand")
        logger.info("2. Motors will gradually increase thrust")
        logger.info("3. Feel when the drone starts to pull upward")
        logger.info("4. Note the thrust value when it feels like it wants to lift")
        logger.info("5. Press Ctrl+C when you feel lift-off force\n")
        
        input("Press Enter to start thrust sweep (Ctrl+C to stop)...")
        
        try:
            # Start from known working value and decrease
            thrust_values = list(range(20000, 50000, 1000))  # 20k to 50k in 1k steps
            
            for thrust in thrust_values:
                logger.info(f"Thrust: {thrust:5d} ({thrust/65535:.3f} normalized, {thrust/2**16:.3f} 0-1 scale)")
                
                # Send thrust command for 1 second
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    cf.commander.send_setpoint(0, 0, 0, thrust)
                    time.sleep(0.01)
                
                # Brief pause between steps
                for _ in range(5):
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.01)
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            logger.info("\n\nCalibration stopped by user!")
        
        finally:
            # Stop motors
            logger.info("Stopping motors...")
            for _ in range(100):
                cf.commander.send_setpoint(0, 0, 0, 0)
                time.sleep(0.01)
            
            logger.info("\n" + "="*60)
            logger.info("CALIBRATION RESULTS")
            logger.info("="*60)
            logger.info("\nRecommended values to update in IsaacLab:")
            logger.info("1. Note the thrust value when drone wanted to lift")
            logger.info("2. Calculate normalized thrust = value / 65535")
            logger.info("3. Update SIMPLEFLIGHT_CFG.mass_props.mass in simpleflight_env_cfg.py")
            logger.info("4. Or adjust thrust_to_weight ratio in simulation")
            logger.info("\nExample:")
            logger.info("  If hover thrust = 35000:")
            logger.info("  - Normalized = 35000/65535 = 0.534")
            logger.info("  - Current sim expects ~0.5 (from action -0.75 + offset)")
            logger.info("  - Adjust mass or thrust_to_weight to match 0.534")
            logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Measure Crazyflie hover thrust")
    parser.add_argument(
        '--uri',
        type=str,
        default='radio://0/80/2M/E7E7E7E7E7',
        help='Crazyflie URI (default: radio://0/80/2M/E7E7E7E7E7)'
    )
    
    args = parser.parse_args()
    measure_hover_thrust(args.uri)


if __name__ == '__main__':
    main()
