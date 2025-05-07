# -*- coding: utf-8 -*-
import time
import threading
# Assuming gpiozero objects (Button, MotionSensor, LED) and config values are passed from main_controller or globally available
# from gpiozero import LED, Button, MotionSensor # Will be handled by main_controller
# import config # Will be imported and used by main_controller, values passed as args

# --- Pedestrian Mode Functions ---
# Note: Global variables like pedestrian_mode_active, active_pedestrian_crossing, etc.,
# and hardware objects like pedestrian_signals, system_lock, stop_event
# will need to be passed as arguments to these functions or managed via a shared state object
# from the main_controller.py for proper modularization.

def activate_pedestrian_mode(crossing_index, shared_state, pedestrian_signals, traffic_signals, config_vars):
    """Activates pedestrian mode for a specific crossing."""
    with shared_state["system_lock"]:
        if shared_state["pedestrian_mode_active"] or shared_state["traffic_locked"] or shared_state["emergency_mode_active"]:
            print(f"🚦 Pedestrian mode for street {crossing_index + 1} cannot be activated now (already active, traffic locked, or emergency).")
            return

        current_time = time.time()
        if current_time < shared_state["pedestrian_cooldown_until"][crossing_index]:
            print(f"⏳ Pedestrian mode for street {crossing_index + 1} in cooldown. Try again in {shared_state["pedestrian_cooldown_until"][crossing_index] - current_time:.1f}s.")
            return

        print(f"🚶 Activating pedestrian mode for Street {crossing_index + 1}...")
        shared_state["traffic_locked"] = True # Lock traffic signals before changing

        # Turn all vehicle traffic to red
        for i in range(config_vars["NUM_STREETS"]):
            traffic_signals[i]["green"].off()
            traffic_signals[i]["red"].on()
            print(f"🚦 Traffic light for Street {i+1} set to RED for pedestrian crossing.")

        # Turn pedestrian signal to green for the requested crossing, red for others
        for i in range(config_vars["NUM_STREETS"]):
            if i == crossing_index:
                pedestrian_signals[i]["red"].off()
                pedestrian_signals[i]["green"].on()
            else:
                pedestrian_signals[i]["green"].off()
                pedestrian_signals[i]["red"].on()

        shared_state["pedestrian_mode_active"] = True
        shared_state["active_pedestrian_crossing"] = crossing_index
        shared_state["pedestrian_mode_start_time"] = current_time
        shared_state["last_motion_time"] = current_time # Assume motion at start
        shared_state["traffic_locked"] = False # Unlock after changes
        print(f"✅ Pedestrian GREEN for Street {crossing_index + 1}. Other streets RED.")

def deactivate_pedestrian_mode(shared_state, pedestrian_signals, config_vars, triggered_by_emergency=False):
    """Deactivates pedestrian mode and restores traffic signals after a delay."""
    with shared_state["system_lock"]:
        if not shared_state["pedestrian_mode_active"]:
            return
        
        crossing_index = shared_state["active_pedestrian_crossing"]
        print(f"🚶 Deactivating pedestrian mode for Street {crossing_index + 1}...")

        pedestrian_signals[crossing_index]["green"].off()
        pedestrian_signals[crossing_index]["red"].on()
        print(f"🚦 Pedestrian signal for Street {crossing_index + 1} set to RED.")

        shared_state["pedestrian_mode_active"] = False
        shared_state["active_pedestrian_crossing"] = -1
        shared_state["pedestrian_cooldown_until"][crossing_index] = time.time() + config_vars["PEDESTRIAN_COOLDOWN_TIME"]
        
        # The main control loop in vehicle_logic.py will handle restoring traffic signals
        # after TRAFFIC_RESTORE_DELAY if not triggered by emergency.
        # If triggered by emergency, emergency_logic handles signals immediately.
        if not triggered_by_emergency:
            shared_state["last_traffic_restore_wait"] = time.time() # Signal vehicle_logic to wait
            print(f"⏳ Traffic signal restoration will be handled by main logic after {config_vars["TRAFFIC_RESTORE_DELAY"]}s delay.")

def monitor_pedestrian_activity(shared_state, pedestrian_signals, config_vars):
    """Monitors pedestrian activity during active pedestrian mode."""
    while not shared_state["stop_event"].is_set():
        if shared_state["pedestrian_mode_active"] and shared_state["active_pedestrian_crossing"] != -1:
            if shared_state["emergency_mode_active"]:
                time.sleep(0.5) # Yield if emergency mode takes over
                continue
            
            current_time = time.time()
            crossing_index = shared_state["active_pedestrian_crossing"]
            elapsed_time = current_time - shared_state["pedestrian_mode_start_time"]
            
            if elapsed_time >= config_vars["MAX_PEDESTRIAN_TIME"]:
                print(f"⏱️ Pedestrian time expired ({config_vars["MAX_PEDESTRIAN_TIME"]} seconds) for street {crossing_index+1}. Stopping pedestrian mode.")
                deactivate_pedestrian_mode(shared_state, pedestrian_signals, config_vars)
                continue
            
            motion_at_active_crossing = shared_state["motion_detected_flags"][crossing_index]
            
            if motion_at_active_crossing:
                shared_state["last_motion_time"] = current_time
            elif current_time - shared_state["last_motion_time"] >= config_vars["MOTION_TIMEOUT"]:
                print(f"🛑 No pedestrian motion for {config_vars["MOTION_TIMEOUT"]} seconds at crossing {crossing_index+1}. Stopping pedestrian mode.")
                deactivate_pedestrian_mode(shared_state, pedestrian_signals, config_vars)
        
        time.sleep(0.5)
    print("🛑 Pedestrian activity monitoring thread stopped.")

def handle_pedestrian_button_press(index, shared_state, pedestrian_signals, traffic_signals, config_vars):
    """Handles button presses for a specific pedestrian crossing."""
    button = pedestrian_signals[index]["button"]
    street_name = f"Street {index + 1}"
    print(f"🔘 Initialized pedestrian button monitoring for {street_name}.")
    
    while not shared_state["stop_event"].is_set():
        try:
            button.wait_for_press(timeout=1) # Timeout to allow checking stop_event
            if shared_state["stop_event"].is_set():
                break
            if not button.is_pressed: # Check if it was a timeout or actual press
                continue

            print(f"🔘 [{street_name}] Button pressed.")
            
            # Debounce or short delay before activation logic
            time.sleep(0.2)
            if shared_state["stop_event"].is_set(): break

            # Check cooldown and other conditions before attempting to activate
            current_time = time.time()
            if current_time < shared_state["pedestrian_cooldown_until"][index]:
                print(f"⏳ Pedestrian mode for {street_name} in cooldown. Try again in {shared_state["pedestrian_cooldown_until"][index] - current_time:.1f}s.")
                continue
            if shared_state["pedestrian_mode_active"] or shared_state["traffic_locked"] or shared_state["emergency_mode_active"]:
                print(f"🚦 Pedestrian mode for {street_name} cannot be activated now (other mode active/locked).")
                continue

            print(f"⏳ Waiting {config_vars["WAIT_AFTER_BUTTON"]} seconds before activating pedestrian mode for {street_name}...")
            wait_end_time = time.time() + config_vars["WAIT_AFTER_BUTTON"]
            while time.time() < wait_end_time and not shared_state["stop_event"].is_set():
                time.sleep(0.1)
            
            if shared_state["stop_event"].is_set(): break
            
            activate_pedestrian_mode(index, shared_state, pedestrian_signals, traffic_signals, config_vars)
            
            # Wait for pedestrian mode to complete or be interrupted
            while shared_state["pedestrian_mode_active"] and shared_state["active_pedestrian_crossing"] == index and not shared_state["emergency_mode_active"] and not shared_state["stop_event"].is_set():
                time.sleep(0.5)
            
            if shared_state["stop_event"].is_set(): break
            
        except Exception as e:
            if shared_state["stop_event"].is_set(): break
            print(f"❌ Error in handle_pedestrian_button_press for {street_name}: {e}")
            time.sleep(1) # Avoid rapid error looping
    print(f"🛑 Pedestrian button monitoring thread for {street_name} stopped.")

def monitor_pir_sensor(index, shared_state, pedestrian_signals):
    """Monitors PIR sensor for a specific crossing."""
    pir = pedestrian_signals[index]["pir"]
    street_name = f"Street {index + 1}"
    print(f"👀 Initialized PIR sensor monitoring for {street_name}.")
    
    def motion_detected_callback():
        if not shared_state["stop_event"].is_set():
            # print(f"PIR {street_name}: Motion DETECTED") # Can be noisy
            shared_state["motion_detected_flags"][index] = True
    
    def motion_stopped_callback():
        if not shared_state["stop_event"].is_set():
            # print(f"PIR {street_name}: Motion STOPPED") # Can be noisy
            shared_state["motion_detected_flags"][index] = False
    
    pir.when_motion = motion_detected_callback
    pir.when_no_motion = motion_stopped_callback
    
    shared_state["stop_event"].wait() # Keep thread alive until stop_event is set
    print(f"🛑 PIR sensor monitoring thread for {street_name} stopped.")


