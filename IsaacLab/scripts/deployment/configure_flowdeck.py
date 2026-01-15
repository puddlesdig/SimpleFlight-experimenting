#!/usr/bin/env python3
"""
Configure Crazyflie for Flow Deck and reset position estimate.
Run this before deploying the policy!
"""

import logging
import time
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_flowdeck(uri):
    """Configure Crazyflie to use Flow Deck for positioning."""
    
    print("\n" + "="*60)
    print("Flow Deck Configuration")
    print("="*60)
    
    cflib.crtp.init_drivers()
    
    print(f"\nConnecting to {uri}...")
    with SyncCrazyflie(uri) as scf:
        cf = scf.cf
        
        print("✓ Connected!")
        time.sleep(1.0)
        
        # Configure estimator
        print("\nConfiguring state estimator...")
        
        # Use Kalman filter (estimator type 2)
        cf.param.set_value('stabilizer.estimator', '2')
        print("  ✓ Set estimator to Kalman (type 2)")
        
        # Check if flow deck is detected
        try:
            flow_deck_status = cf.param.get_value('deck.bcFlow2')
            print(f"  ✓ Flow Deck detected: {flow_deck_status}")
        except:
            print("  ⚠️  Flow Deck not detected - check connection")
        
        # Reset estimator
        print("\nResetting position estimate...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.5)
        cf.param.set_value('kalman.resetEstimation', '0')
        print("  ✓ Position reset")
        
        time.sleep(1.0)
        
        # Read position
        from cflib.crazyflie.log import LogConfig
        
        position = [0, 0, 0]
        
        def pos_callback(timestamp, data, logconf):
            position[0] = data['stateEstimate.x']
            position[1] = data['stateEstimate.y']
            position[2] = data['stateEstimate.z']
        
        log_pos = LogConfig(name='Position', period_in_ms=100)
        log_pos.add_variable('stateEstimate.x', 'float')
        log_pos.add_variable('stateEstimate.y', 'float')
        log_pos.add_variable('stateEstimate.z', 'float')
        
        cf.log.add_config(log_pos)
        log_pos.data_received_cb.add_callback(pos_callback)
        log_pos.start()
        
        time.sleep(2.0)
        
        print(f"\nCurrent position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
        
        if abs(position[2]) < 0.1:
            print("✓ Position looks good (close to ground)")
        else:
            print("⚠️  Z position not zero - make sure drone is on flat surface")
        
        log_pos.stop()
        
        print("\n" + "="*60)
        print("✅ Flow Deck Configured!")
        print("="*60)
        print("\nIMPORTANT:")
        print("1. Place drone on TEXTURED surface (not blank/shiny)")
        print("2. Flow deck needs ~0.1-2m altitude to work")
        print("3. Keep drone level during takeoff")
        print("\nReady to deploy policy!")
        print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure Flow Deck")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E7",
                        help="Crazyflie URI")
    args = parser.parse_args()
    
    configure_flowdeck(args.uri)
