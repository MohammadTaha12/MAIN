# -*- coding: utf-8 -*-
import cv2
import time
import threading
import queue
import json
import os
import numpy as np
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO
from gpiozero import LED, Button, MotionSensor

# Import aplication modules
import config
import pedestrian_logic
import emergency_logic
import vehicle_logic

# --- Microphone Management Class (kept in main_controller for now) ---
class MicrophoneManager:
    def __init__(self, num_streets, mic_mapping_file):
        self.p = pyaudio.PyAudio()
        self.mic_indices = []
        self.mapping = {}
        self.street_names = [f"شارع {i+1}" for i in range(num_streets)]
        self.num_streets = num_streets
        self.mic_mapping_file = mic_mapping_file
        # self.load_mapping() # Call this explicitly after creation

    def get_available_mics(self):
        available_mics = []
        # print("\nSearching for audio input devices...")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                available_mics.append({
                    "index": info["index"],
                    "name": info["name"],
                    "channels": info["maxInputChannels"]
                })
        return available_mics

    def print_available_mics(self):
        print("="*50)
        print("Available audio input devices:")
        print("="*50)
        available_mics = self.get_available_mics()
        if not available_mics:
            print("❌ No audio input devices found!")
            return
        for mic in available_mics:
            print(f"Index {mic["index"]}: {mic["name"]} (Channels: {mic["channels"]})")
        print("="*50)

    def load_mapping(self):
        if os.path.exists(self.mic_mapping_file):
            try:
                with open(self.mic_mapping_file, "r") as f:
                    self.mapping = json.load(f)
                self.mapping = {int(k): v for k, v in self.mapping.items()} # Ensure keys are int
                self.mic_indices = list(self.mapping.keys())
                mapped_street_indices = set(self.mapping.values())
                required_street_indices = set(range(self.num_streets))
                if len(self.mic_indices) == self.num_streets and mapped_street_indices == required_street_indices:
                    print(f"✅ Loaded mapping from {self.mic_mapping_file}: {self.mapping}")
                    return True
                else:
                    print(f"⚠️ Mapping file \'{self.mic_mapping_file}\' does not match number of streets ({self.num_streets}). Resetting.")
                    self.mapping = {}
                    self.mic_indices = []
            except Exception as e:
                print(f"❌ Error loading mapping file: {e}. Resetting.")
                self.mapping = {}
                self.mic_indices = []
        return False

    def save_mapping(self):
        try:
            with open(self.mic_mapping_file, "w") as f:
                json.dump(self.mapping, f)
            print(f"✅ Saved microphone mapping to {self.mic_mapping_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving mapping file: {e}")
            return False

    def setup_mapping(self):
        self.print_available_mics()
        available_mics = self.get_available_mics()
        if not available_mics:
            print("❌ Cannot setup mapping: No audio input devices found!")
            return False
        if len(available_mics) < self.num_streets:
            print(f"❌ Cannot setup mapping: Available microphones ({len(available_mics)}) less than number of streets ({self.num_streets})!")
            return False

        print(f"\n--- Setting up microphone mapping for {self.num_streets} streets ---")
        self.mapping = {}
        self.mic_indices = []
        assigned_indices = set()
        for i in range(self.num_streets):
            street_name = self.street_names[i]
            while True:
                try:
                    print(f"\nStreet: {street_name}")
                    index_str = input(f"Enter microphone index for {street_name}: ")
                    index = int(index_str)
                    if not any(mic["index"] == index for mic in available_mics):
                        print(f"❌ Error: Index {index} is invalid.")
                        continue
                    if index in assigned_indices:
                        print(f"❌ Error: Index {index} already assigned.")
                        continue
                    self.mic_indices.append(index)
                    self.mapping[index] = i
                    assigned_indices.add(index)
                    print(f"✅ Assigned microphone {index} to {street_name} (Street index: {i})")
                    break
                except ValueError:
                    print("❌ Error: Please enter a valid number.")
        self.save_mapping()
        print("\n✅ Microphone mapping setup completed.")
        return True

    def get_street_index_for_mic(self, mic_index):
        return self.mapping.get(mic_index, -1)

    def get_mic_indices(self):
        return self.mic_indices

    def terminate(self):
        self.p.terminate()
        print("🎤 PyAudio terminated.")

# --- Global Shared State ---
# This dictionary will hold all state variables, events, locks, and pre-loaded models
# to be passed to various logic modules and threads.
shared_state = {
    # System Control
    "system_lock": threading.Lock(),
    "stop_event": threading.Event(),

    # Pedestrian Mode
    "traffic_locked": False, # True when pedestrian/emergency mode is changing signals
    "pedestrian_mode_active": False,
    "pedestrian_mode_start_time": 0,
    "last_motion_time": 0,
    "motion_detected_flags": [False] * config.NUM_STREETS,
    "active_pedestrian_crossing": -1,
    "pedestrian_cooldown_until": [0] * config.NUM_STREETS,

    # Emergency Mode
    "emergency_mode_active": False,
    "emergency_street_index": -1,
    "emergency_mode_start_time": 0,
    "last_emergency_trigger_time": 0, # Cooldown for emergency triggers

    # Traffic Light State (for vehicle_logic)
    "previous_green": None,
    "green_start_time": 0,
    "green_reason": None,
    "green_lock_until": 0,
    "all_streets_empty": False,
    "last_traffic_restore_wait": 0, # Timestamp when a restore delay started

    # Traffic Analysis State (for vehicle_logic)
    "last_process_time": 0, # For camera processing interval
    "last_wait_update": 0, # For updating wait times every second
    "wait_times": [0] * config.NUM_STREETS,
    "wait_start_time": [0] * config.NUM_STREETS,
    "previous_counts": [0] * config.NUM_STREETS,
    "empty_since": [0] * config.NUM_STREETS,

    # Data Queues
    "audio_data_queue": queue.Queue(),

    # Pre-loaded Models
    "yolo_model": None,
    "yamnet_model": None,
    "yamnet_class_names": [],
}

# --- Hardware Objects ---
traffic_signals_hw = []
pedestrian_signals_hw = []
cameras_hw = []
mic_manager_obj = None

# --- Utility Functions ---
def initialize_system_signals():
    print("Initializing signal states (All Red)...")
    for i in range(config.NUM_STREETS):
        traffic_signals_hw[i]["red"].on()
        traffic_signals_hw[i]["green"].off()
        pedestrian_signals_hw[i]["red"].on()
        pedestrian_signals_hw[i]["green"].off()
    print("✅ Signals initialized.")

# --- Audio Capture Thread Function ---
def audio_capture_thread_worker(device_index, shared_state_ref, mic_manager_ref, config_ref):
    p_audio = pyaudio.PyAudio() # Each thread needs its own PyAudio instance
    stream = None
    street_idx = mic_manager_ref.get_street_index_for_mic(device_index)
    street_name_log = f"Street {street_idx + 1}" if street_idx != -1 else f"Unknown Mic {device_index}"
    
    try:
        stream = p_audio.open(format=config_ref.AUDIO_FORMAT,
                              channels=config_ref.AUDIO_CHANNELS,
                              rate=config_ref.AUDIO_RATE,
                              input=True,
                              frames_per_buffer=config_ref.AUDIO_CHUNK,
                              input_device_index=device_index)
        print(f"🎤 Started audio capture thread for {street_name_log} (Mic Index: {device_index})...")
        
        while not shared_state_ref["stop_event"].is_set():
            try:
                data = stream.read(config_ref.AUDIO_CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                shared_state_ref["audio_data_queue"].put((audio_np, device_index))
            except IOError as e_io:
                if not shared_state_ref["stop_event"].is_set(): # Log only if not stopping
                    print(f"❌ Audio IO Error in {street_name_log} (Mic: {device_index}): {e_io}")
                    time.sleep(1) # Brief pause before retrying
            except Exception as e_cap:
                if not shared_state_ref["stop_event"].is_set():
                    print(f"❌ Unexpected Error in audio capture for {street_name_log} (Mic: {device_index}): {e_cap}")
                    break # Exit thread on unexpected error

    except Exception as e_stream:
        print(f"❌ Failed to start audio stream for {street_name_log} (Mic: {device_index}): {e_stream}")
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e_close:
                print(f"❌ Error closing audio stream for {street_name_log} (Mic: {device_index}): {e_close}")
        p_audio.terminate()
        print(f"🛑 Stopped audio capture thread for {street_name_log} (Mic: {device_index}).")

# --- Main Audio Processing Loop (runs in a dedicated thread) ---
def audio_processing_loop(shared_state_ref, mic_manager_ref, traffic_signals_ref, pedestrian_signals_ref, config_ref):
    print("🎧 Starting Audio Processing Loop...")
    try:
        while not shared_state_ref["stop_event"].is_set():
            try:
                audio_np, mic_idx = shared_state_ref["audio_data_queue"].get(block=True, timeout=0.5) # Timeout to check stop_event
                emergency_logic.analyze_audio_with_yamnet(audio_np, mic_idx, shared_state_ref, mic_manager_ref, traffic_signals_ref, pedestrian_signals_ref, config_ref)
                shared_state_ref["audio_data_queue"].task_done()
            except queue.Empty:
                continue # Normal if queue is empty, just check stop_event and loop
            except Exception as e_proc:
                if not shared_state_ref["stop_event"].is_set():
                    print(f"❌ Error in audio processing loop: {e_proc}")
                    time.sleep(1) # Avoid rapid error looping
    except Exception as e_outer:
         print(f"❌ CRITICAL Error in audio_processing_loop outer try: {e_outer}")
    finally:
        print("🛑 Audio processing loop stopped.")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🚦 Starting Integrated Smart Traffic Light System (Modular)...")
    
    # Convert config module to a dictionary for easier passing
    config_vars = {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
    shared_state["config_vars"] = config_vars # Make config accessible within shared_state if needed by logic modules

    # --- Initialize Hardware and Models ---
    print("\nLoading YOLO model...")
    try:
        shared_state["yolo_model"] = YOLO(config.YOLO_MODEL_PATH)
        print("✅ YOLO model loaded.")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        exit()

    print("\nLoading YAMNet model...")
    try:
        shared_state["yamnet_model"] = hub.load(config.YAMNET_MODEL_HANDLE)
        class_map_path = shared_state["yamnet_model"].class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            for row in f:
                shared_state["yamnet_class_names"].append(row.strip().split(",")[2])
        print("✅ YAMNet model and class names loaded.")
    except Exception as e:
        print(f"❌ Failed to load YAMNet model: {e}")
        exit()

    print("\nInitializing cameras...")
    for i, cam_idx_str in enumerate(config.CAMERA_INDICES):
        try:
            # Attempt to convert to int if it's a device number, otherwise use as string (path)
            try: cam_val = int(cam_idx_str) 
            except ValueError: cam_val = cam_idx_str
            
            cam = cv2.VideoCapture(cam_val)
            if not cam.isOpened():
                raise Exception(f"Camera {i+1} at {cam_idx_str} failed to open.")
            cameras_hw.append(cam)
            print(f"✅ Camera {i+1} ({cam_idx_str}) initialized.")
        except Exception as e:
            print(f"❌ {e}")
            for c_hw in cameras_hw: c_hw.release()
            exit()

    print("\nInitializing GPIO devices...")
    try:
        for i in range(config.NUM_STREETS):
            traffic_signals_hw.append({
                "red": LED(config.TRAFFIC_SIGNAL_PINS[i]["red"]),
                "green": LED(config.TRAFFIC_SIGNAL_PINS[i]["green"])
            })
            pedestrian_signals_hw.append({
                "red": LED(config.PEDESTRIAN_SIGNAL_PINS[i]["red"]),
                "green": LED(config.PEDESTRIAN_SIGNAL_PINS[i]["green"]),
                "button": Button(config.PEDESTRIAN_SIGNAL_PINS[i]["button"], pull_up=False),
                "pir": MotionSensor(config.PEDESTRIAN_SIGNAL_PINS[i]["pir"], queue_len=1, sample_rate=10, threshold=0.2)
            })
            print(f"✅ GPIO devices for Street {i+1} initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize GPIO devices: {e}")
        exit()
    
    initialize_system_signals() # Set all to RED initially

    print("\nLoading microphone mapping...")
    mic_manager_obj = MicrophoneManager(config.NUM_STREETS, config.MIC_MAPPING_FILE)
    if not mic_manager_obj.load_mapping():
        print(f"❌ Error: Microphone mapping file \'{config.MIC_MAPPING_FILE}\' not found or invalid.")
        print("Please ensure the mapping file exists or run with an option to configure it.")
        # mic_manager_obj.setup_mapping() # Optionally run setup if load fails
        # if not mic_manager_obj.load_mapping(): exit(1) # Exit if setup also fails or isn't run
        exit(1)

    # --- Start All Threads ---
    threads_list = []
    print("\nStarting Worker Threads...")

    # Audio Capture Threads
    for mic_idx_loop in mic_manager_obj.get_mic_indices():
        thread = threading.Thread(target=audio_capture_thread_worker, 
                                  args=(mic_idx_loop, shared_state, mic_manager_obj, config), daemon=True)
        threads_list.append(thread)
        thread.start()

    # Audio Processing Thread
    audio_proc_thread = threading.Thread(target=audio_processing_loop, 
                                         args=(shared_state, mic_manager_obj, traffic_signals_hw, pedestrian_signals_hw, config_vars), daemon=True)
    threads_list.append(audio_proc_thread)
    audio_proc_thread.start()

    # Pedestrian Button Threads
    for i in range(config.NUM_STREETS):
        thread = threading.Thread(target=pedestrian_logic.handle_pedestrian_button_press, 
                                  args=(i, shared_state, pedestrian_signals_hw, traffic_signals_hw, config_vars), daemon=True)
        threads_list.append(thread)
        thread.start()

    # PIR Sensor Threads
    for i in range(config.NUM_STREETS):
        thread = threading.Thread(target=pedestrian_logic.monitor_pir_sensor, 
                                  args=(i, shared_state, pedestrian_signals_hw), daemon=True)
        threads_list.append(thread)
        thread.start()

    # Pedestrian Activity Monitoring Thread
    ped_activity_thread = threading.Thread(target=pedestrian_logic.monitor_pedestrian_activity, 
                                           args=(shared_state, pedestrian_signals_hw, config_vars), daemon=True)
    threads_list.append(ped_activity_thread)
    ped_activity_thread.start()

    # Main Vehicle Traffic Control Thread
    vehicle_control_thread = threading.Thread(target=vehicle_logic.control_traffic_signals, 
                                              args=(shared_state, traffic_signals_hw, cameras_hw, config_vars))
    threads_list.append(vehicle_control_thread)
    vehicle_control_thread.start()

    print("\n✅ All threads started. System is running.")

    # --- Keep Main Thread Alive & Handle Shutdown ---
    try:
        while not shared_state["stop_event"].is_set():
            # The main thread can perform other checks or simply wait
            # For example, check if the critical vehicle_control_thread is alive
            if not vehicle_control_thread.is_alive() and not shared_state["stop_event"].is_set():
                print("❌ CRITICAL: Vehicle control thread died unexpectedly! Stopping system.")
                shared_state["stop_event"].set()
                break
            time.sleep(1) # Check every second
            
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard Interrupt detected. Stopping system...")
    except Exception as e_main:
        print(f"❌ Critical error in main controller: {e_main}. Stopping system...")
    finally:
        print("\nInitiating shutdown sequence...")
        shared_state["stop_event"].set()
        
        print("Joining threads...")
        for t in threads_list:
            if t.is_alive():
                t.join(timeout=5.0) # Wait for threads to finish
                if t.is_alive():
                    print(f"⚠️ Thread {t.name} did not terminate gracefully.")
        
        print("🧹 Cleaning up resources...")
        if mic_manager_obj:
            mic_manager_obj.terminate()
        for cam_hw in cameras_hw:
            cam_hw.release()
        cv2.destroyAllWindows()
        # GPIO cleanup is usually handled by gpiozero on exit, but explicit off can be added if needed
        # for i in range(config.NUM_STREETS):
        #     traffic_signals_hw[i]["red"].off()
        #     traffic_signals_hw[i]["green"].off()
        #     pedestrian_signals_hw[i]["red"].off()
        #     pedestrian_signals_hw[i]["green"].off()
        print("🛑 System stopped.")

