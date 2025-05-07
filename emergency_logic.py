# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow_hub as hub # For YAMNet, will be loaded by main_controller
import tensorflow as tf # For YAMNet, will be loaded by main_controller
# import config # Imported by main_controller, values passed as args
# Global variables like emergency_mode_active, emergency_street_index, etc.,
# and hardware objects like traffic_signals, pedestrian_signals, system_lock, stop_event, yamnet_model, yamnet_class_names
# will need to be passed as arguments or managed via a shared state object from main_controller.py

def analyze_audio_with_yamnet(waveform, mic_device_index, shared_state, mic_manager, traffic_signals, pedestrian_signals, config_vars):
    """Analyzes audio waveform using YAMNet to detect emergency sounds."""
    try:
        street_index = mic_manager.get_street_index_for_mic(mic_device_index)
        if street_index == -1:
            # print(f"⚠️ Warning: Received audio from unmapped mic index {mic_device_index}") # Can be noisy
            return

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32) / 32768.0 # Normalize to [-1.0, 1.0]

        # Ensure waveform is of the correct length for YAMNet
        if len(waveform) < config_vars["YAMNET_WINDOW_SAMPLES"]:
            waveform = np.pad(waveform, (0, config_vars["YAMNET_WINDOW_SAMPLES"] - len(waveform)), "constant")
        elif len(waveform) > config_vars["YAMNET_WINDOW_SAMPLES"]:
            waveform = waveform[:config_vars["YAMNET_WINDOW_SAMPLES"]]

        # YAMNet model and class names should be loaded once in main_controller and passed here
        yamnet_model = shared_state["yamnet_model"]
        yamnet_class_names = shared_state["yamnet_class_names"]

        scores, embeddings, log_mel_spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()[0] # Get the first (and only) batch

        detected_emergency = False
        detected_class = "None"
        max_score = 0.0

        for i, score in enumerate(scores_np):
            class_name = yamnet_class_names[i]
            # Check if any emergency keyword is part of the detected class name (case-insensitive)
            is_emergency_sound = any(keyword in class_name.lower() for keyword in config_vars["EMERGENCY_KEYWORDS"])
            if is_emergency_sound and score > config_vars["EMERGENCY_DETECTION_THRESHOLD"]:
                if score > max_score:
                    detected_emergency = True
                    detected_class = class_name
                    max_score = score
        
        if detected_emergency:
            print(f"🚨 EMERGENCY SOUND DETECTED in Street {street_index + 1} (Mic: {mic_device_index}) - Sound: {detected_class}, Score: {max_score:.2f}")
            trigger_emergency_mode(street_index, shared_state, traffic_signals, pedestrian_signals, config_vars)

    except Exception as e:
        print(f"❌ Error during YAMNet analysis for mic {mic_device_index}: {e}")

def trigger_emergency_mode(street_index, shared_state, traffic_signals, pedestrian_signals, config_vars):
    """Activates emergency mode for the specified street."""
    current_time = time.time()
    
    # Cooldown to prevent rapid re-triggering or overlapping emergencies
    if current_time - shared_state.get("last_emergency_trigger_time", 0) < config_vars["EMERGENCY_MODE_DURATION"]:
        # print(f"🚫 Ignoring emergency from Street {street_index + 1} - Another emergency active or cooling down.")
        return

    with shared_state["system_lock"]:
        if shared_state["emergency_mode_active"] and shared_state["emergency_street_index"] == street_index:
            # print(f"ℹ️ Emergency mode already active for Street {street_index + 1}")
            shared_state["emergency_mode_start_time"] = current_time # Extend duration if re-triggered for same street
            return
        elif shared_state["emergency_mode_active"]:
            print(f"⚠️ Emergency already active for Street {shared_state['emergency_street_index']+1}, cannot trigger for Street {street_index+1} yet.")
            return

        print(f"⚡️ TRIGGERING EMERGENCY MODE for Street {street_index + 1} ⚡️")
        shared_state["emergency_mode_active"] = True
        shared_state["emergency_street_index"] = street_index
        shared_state["emergency_mode_start_time"] = current_time
        shared_state["last_emergency_trigger_time"] = current_time

        # If pedestrian mode was active, deactivate it immediately
        if shared_state["pedestrian_mode_active"]:
            print("🚶 Emergency overrides pedestrian mode. Deactivating pedestrian mode.")
            # Need to call deactivate_pedestrian_mode from pedestrian_logic.py
            # This creates a dependency. For now, we'll replicate basic deactivation.
            active_ped_crossing = shared_state["active_pedestrian_crossing"]
            if active_ped_crossing != -1:
                 pedestrian_signals[active_ped_crossing]["green"].off()
                 pedestrian_signals[active_ped_crossing]["red"].on()
            shared_state["pedestrian_mode_active"] = False
            shared_state["active_pedestrian_crossing"] = -1
            # Consider calling a more complete deactivation function from pedestrian_logic if available

        # Set all pedestrian signals to RED
        for i in range(config_vars["NUM_STREETS"]):
            pedestrian_signals[i]["green"].off()
            pedestrian_signals[i]["red"].on()

        # Set traffic signals for emergency
        for i in range(config_vars["NUM_STREETS"]):
            if i == street_index: # Emergency street gets GREEN
                traffic_signals[i]["red"].off()
                traffic_signals[i]["green"].on()
            else: # Other streets get RED
                traffic_signals[i]["green"].off()
                traffic_signals[i]["red"].on()
            print(f"🚦 Emergency: {"GREEN" if i == street_index else "RED"} for Street {i+1}")

def deactivate_emergency_mode(shared_state, config_vars):
    """Deactivates emergency mode."""
    with shared_state["system_lock"]:
        if not shared_state["emergency_mode_active"]:
            return
        
        street_idx = shared_state["emergency_street_index"]
        print(f"✅ Deactivating Emergency Mode for Street {street_idx + 1}.")
        shared_state["emergency_mode_active"] = False
        shared_state["emergency_street_index"] = -1
        
        # The main control loop in vehicle_logic.py will handle restoring traffic signals
        # after TRAFFIC_RESTORE_DELAY.
        shared_state["last_traffic_restore_wait"] = time.time() # Signal vehicle_logic to wait
        print(f"⏳ Traffic signal restoration will be handled by main logic after {config_vars["TRAFFIC_RESTORE_DELAY"]}s delay.")


