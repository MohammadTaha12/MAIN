# -*- coding: utf-8 -*-
import time
# import cv2 # YOLO model and camera objects will be handled by main_controller and frames passed
# from ultralytics import YOLO # YOLO model will be loaded by main_controller and passed
# import config # Imported by main_controller, values passed as args
# Global variables (like previous_green, green_start_time, etc.) and hardware/model objects
# (traffic_signals, cameras, yolo_model, system_lock, stop_event) will need to be passed as arguments
# or managed via a shared state object from main_controller.py.

def count_vehicles(results, config_vars):
    """Counts vehicles from YOLO results based on configured classes."""
    count = 0
    for result in results: # results is expected to be a list of YOLO results, one per frame/camera
        for box in result.boxes:
            if int(box.cls[0]) in config_vars["VEHICLE_CLASSES"]:
                count += 1
    return count

def control_traffic_signals(shared_state, traffic_signals, cameras, config_vars):
    """Main logic for controlling traffic signals based on vehicle detection."""
    print("🚦 Vehicle logic control thread started.")
    yolo_model = shared_state["yolo_model"] # Get pre-loaded model

    while not shared_state["stop_event"].is_set():
        try:
            current_time = time.time()
            
            # --- Priority 1: Emergency Mode Check ---
            if shared_state["emergency_mode_active"]:
                if current_time - shared_state["emergency_mode_start_time"] >= config_vars["EMERGENCY_MODE_DURATION"]:
                    print(f"⏱️ Emergency mode duration ({config_vars["EMERGENCY_MODE_DURATION"]} seconds) expired in vehicle_logic.")
                    # Deactivation is now handled by emergency_logic, which sets last_traffic_restore_wait
                    # emergency_logic.deactivate_emergency_mode(shared_state, config_vars) # This was a circular call, bad idea
                    # Instead, emergency_logic.py should manage its own deactivation timer or be called by main controller to check
                    # For now, assume emergency_logic handles its own timeout or main_controller polls it.
                    # This control_traffic_signals loop just yields if emergency is active.
                    pass # Emergency logic handles its own deactivation and signal restoration trigger
                time.sleep(0.5) # Yield to other threads if emergency is active
                continue

            # --- Priority 2: Pedestrian Mode Check ---
            if shared_state["pedestrian_mode_active"] or shared_state["traffic_locked"]:
                # Pedestrian logic handles its own signals and traffic locking
                time.sleep(0.5) # Yield to other threads
                continue

            # --- Restore delay after pedestrian/emergency mode ---
            if shared_state.get("last_traffic_restore_wait", 0) > 0:
                if current_time - shared_state["last_traffic_restore_wait"] >= config_vars["TRAFFIC_RESTORE_DELAY"]:
                    print("✅ Restore delay finished in vehicle_logic. Resuming normal traffic control.")
                    shared_state["last_traffic_restore_wait"] = 0
                    # Initialize signals and state (this should ideally be a separate function in main_controller or this module)
                    for i in range(config_vars["NUM_STREETS"]):
                        traffic_signals[i]["red"].on()
                        traffic_signals[i]["green"].off()
                    shared_state["previous_green"] = None
                    shared_state["green_reason"] = None
                    shared_state["green_lock_until"] = 0
                    shared_state["wait_times"] = [0] * config_vars["NUM_STREETS"]
                    shared_state["wait_start_time"] = [0] * config_vars["NUM_STREETS"]
                    shared_state["previous_counts"] = [0] * config_vars["NUM_STREETS"]
                    shared_state["empty_since"] = [0] * config_vars["NUM_STREETS"]
                    shared_state["all_streets_empty"] = False
                    shared_state["last_process_time"] = 0 # Force immediate processing
                else:
                    time.sleep(0.5) # Still in restore delay
                    continue

            # --- Wait times update (every second) ---
            if current_time - shared_state.get("last_wait_update", 0) >= 1:
                for i in range(config_vars["NUM_STREETS"]):
                    is_waiting = (shared_state["previous_green"] is None or shared_state["previous_green"] != i)
                    if is_waiting:
                        if shared_state["previous_counts"][i] > 0 and shared_state["wait_start_time"][i] == 0:
                            shared_state["wait_start_time"][i] = current_time
                            shared_state["wait_times"][i] = 0
                        elif shared_state["previous_counts"][i] > 0 and shared_state["wait_start_time"][i] > 0:
                            shared_state["wait_times"][i] = int(current_time - shared_state["wait_start_time"][i])
                        elif shared_state["previous_counts"][i] == 0:
                            shared_state["wait_start_time"][i] = 0
                            shared_state["wait_times"][i] = 0
                    else: # Street is green
                        shared_state["wait_start_time"][i] = 0
                        shared_state["wait_times"][i] = 0
                shared_state["last_wait_update"] = current_time

            # --- Process cameras and YOLO at TRAFFIC_SCAN_INTERVAL ---
            if current_time - shared_state.get("last_process_time", 0) >= config_vars["TRAFFIC_SCAN_INTERVAL"]:
                shared_state["last_process_time"] = current_time
                frames = []
                valid_reads = True

                for i in range(config_vars["NUM_STREETS"]):
                    ret, frame = cameras[i].read() # cameras are cv2.VideoCapture objects
                    if not ret:
                        print(f"⚠️ Problem reading camera {i+1} in vehicle_logic")
                        valid_reads = False
                        break
                    frames.append(frame)

                if not valid_reads:
                    time.sleep(1) # Wait a bit before retrying camera reads
                    continue

                try:
                    # YOLO model expects a list of frames
                    yolo_results_list = yolo_model(frames, verbose=False) # verbose=False to reduce console spam
                    # Ensure yolo_results_list has one result per frame
                    if len(yolo_results_list) != len(frames):
                        print(f"❌ YOLO processing error: Mismatch in results and frames count.")
                        continue
                    current_vehicle_counts = [count_vehicles([yolo_results_list[i]], config_vars) for i in range(config_vars["NUM_STREETS"])]
                except Exception as e:
                    print(f"❌ Error during YOLO processing in vehicle_logic: {e}")
                    continue # Skip this cycle if YOLO fails

                # Update empty_since times
                for i in range(config_vars["NUM_STREETS"]):
                    if shared_state["previous_counts"][i] > 0 and current_vehicle_counts[i] == 0:
                        if shared_state["empty_since"][i] == 0: # If it just became empty
                            shared_state["empty_since"][i] = current_time
                    elif current_vehicle_counts[i] > 0: # If there are vehicles, it's not empty
                        shared_state["empty_since"][i] = 0
                
                shared_state["previous_counts"] = current_vehicle_counts.copy()

                # Handle all streets empty scenario
                if all(count == 0 for count in current_vehicle_counts):
                    if not shared_state["all_streets_empty"]:
                        print("⚠️ All streets are empty, setting all traffic signals to RED (vehicle_logic)")
                        shared_state["all_streets_empty"] = True
                        if shared_state["previous_green"] is not None:
                            with shared_state["system_lock"]:
                                traffic_signals[shared_state["previous_green"]]["green"].off()
                                traffic_signals[shared_state["previous_green"]]["red"].on()
                                shared_state["previous_green"] = None
                                shared_state["green_reason"] = None
                                shared_state["green_lock_until"] = 0
                    continue # No need to decide on green if all empty
                else:
                    shared_state["all_streets_empty"] = False

                # --- Traffic Light Decision Logic ---
                force_change_current_green = False
                can_change_green = shared_state["green_lock_until"] <= current_time
                next_priority_street = -1
                next_green_reason_for_log = None

                # Check if current green street needs to change
                if shared_state["previous_green"] is not None and can_change_green:
                    pg_idx = shared_state["previous_green"]
                    # Condition 1: Current green street is empty for a while
                    if current_vehicle_counts[pg_idx] == 0 and shared_state["empty_since"][pg_idx] > 0 and \
                       (current_time - shared_state["empty_since"][pg_idx] >= config_vars["EMPTY_VERIFICATION_TIME"]):
                        print(f"🚫 Street {pg_idx+1} (current green) confirmed empty, forcing signal change.")
                        force_change_current_green = True
                    # Condition 2: Current green was for long_wait and its duration ended
                    elif shared_state["green_reason"] == "long_wait" and \
                         (current_time - shared_state["green_start_time"] >= config_vars["LONG_WAIT_GREEN_DURATION"]):
                        print(f"⏱️ Green duration ended for Street {pg_idx+1} (long wait). Forcing change.")
                        force_change_current_green = True
                
                # Determine next green street if allowed or forced
                if can_change_green or force_change_current_green:
                    # Priority 1: Longest waiting street with vehicles
                    long_wait_streets_indices = [
                        i for i in range(config_vars["NUM_STREETS"])
                        if shared_state["wait_times"][i] >= config_vars["LONG_WAIT_THRESHOLD"] and current_vehicle_counts[i] > 0
                    ]
                    if long_wait_streets_indices:
                        next_priority_street = max(long_wait_streets_indices, key=lambda i: shared_state["wait_times"][i])
                        next_green_reason_for_log = "long_wait"
                    else:
                        # Priority 2: Street with most vehicles (not current green unless forced or no other choice)
                        streets_with_vehicles_not_green = [
                            i for i in range(config_vars["NUM_STREETS"])
                            if current_vehicle_counts[i] > 0 and i != shared_state["previous_green"]
                        ]
                        if streets_with_vehicles_not_green:
                            next_priority_street = max(streets_with_vehicles_not_green, key=lambda i: current_vehicle_counts[i])
                            next_green_reason_for_log = "high_count"
                        elif shared_state["previous_green"] is not None and current_vehicle_counts[shared_state["previous_green"]] > 0 and not force_change_current_green:
                            next_priority_street = shared_state["previous_green"] # Keep current green if it still has cars and no other priority
                            next_green_reason_for_log = shared_state["green_reason"]
                        elif force_change_current_green: # If forced and no other street has cars, turn current red
                             next_priority_street = -1 # This will turn current green to red, and no new green
                             next_green_reason_for_log = "forced_empty"

                # --- Apply Signal Change ---
                if (next_priority_street != shared_state["previous_green"] and next_priority_street != -1 and can_change_green) or \
                   (force_change_current_green and next_priority_street != shared_state["previous_green"]):
                    with shared_state["system_lock"]:
                        # Turn off previous green
                        if shared_state["previous_green"] is not None:
                            pg_idx_old = shared_state["previous_green"]
                            traffic_signals[pg_idx_old]["green"].off()
                            traffic_signals[pg_idx_old]["red"].on()
                            print(f"🚦 Street {pg_idx_old+1} set to RED.")
                            shared_state["wait_times"][pg_idx_old] = 0 # Reset wait time as it was green
                            shared_state["wait_start_time"][pg_idx_old] = 0
                        
                        # Turn on new green (if any)
                        if next_priority_street != -1:
                            traffic_signals[next_priority_street]["red"].off()
                            traffic_signals[next_priority_street]["green"].on()
                            print(f"🚦 Street {next_priority_street+1} set to GREEN (Reason: {next_green_reason_for_log}).")
                            shared_state["previous_green"] = next_priority_street
                            shared_state["green_start_time"] = current_time
                            shared_state["green_reason"] = next_green_reason_for_log
                            shared_state["wait_times"][next_priority_street] = 0 # Reset wait time for new green
                            shared_state["wait_start_time"][next_priority_street] = 0
                            shared_state["empty_since"][next_priority_street] = 0 # Reset empty since for new green
                            if next_green_reason_for_log == "long_wait":
                                shared_state["green_lock_until"] = current_time + config_vars["LONG_WAIT_GREEN_DURATION"]
                            else:
                                shared_state["green_lock_until"] = 0 # No specific lock duration for high_count, relies on EMPTY_VERIFICATION_TIME
                        else: # No street gets green (e.g., current green forced empty, no other cars)
                            shared_state["previous_green"] = None
                            shared_state["green_reason"] = None
                            shared_state["green_lock_until"] = 0
                elif force_change_current_green and next_priority_street == -1 and shared_state["previous_green"] is not None:
                     # This case handles when current green is forced off and no other street is eligible for green.
                    with shared_state["system_lock"]:
                        pg_idx_old = shared_state["previous_green"]
                        traffic_signals[pg_idx_old]["green"].off()
                        traffic_signals[pg_idx_old]["red"].on()
                        print(f"🚦 Street {pg_idx_old+1} (forced empty) set to RED. No new green assigned.")
                        shared_state["wait_times"][pg_idx_old] = 0
                        shared_state["wait_start_time"][pg_idx_old] = 0
                        shared_state["previous_green"] = None
                        shared_state["green_reason"] = None
                        shared_state["green_lock_until"] = 0

            time.sleep(0.1) # Main loop sleep

        except Exception as e:
            print(f"❌ Error in control_traffic_signals (vehicle_logic): {e}")
            shared_state["stop_event"].set() # Critical error, stop the system
            break
    print("🛑 Vehicle logic control thread stopped.")

