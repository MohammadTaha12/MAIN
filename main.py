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

# --- Configuration Constants ---
# YOLO
YOLO_MODEL_PATH = "yolov5s.pt"
VEHICLE_CLASSES = [2, 5, 7]  # YOLO classes for car, bus, truck

# Cameras
CAMERA_INDICES = ["/dev/video0", "/dev/video2", "/dev/video4"]
NUM_STREETS = len(CAMERA_INDICES)

# GPIO Pins
TRAFFIC_SIGNAL_PINS = [
    {"red": 23, "green": 24},  # Street 1
    {"red": 15, "green": 14},  # Street 2
    {"red": 21, "green": 20}   # Street 3
]

PEDESTRIAN_SIGNAL_PINS = [
    {"red": 26, "green": 25, "button": 11, "pir": 22}, # Street 1
    {"red": 16, "green": 12, "button": 9,  "pir": 27}, # Street 2
    {"red": 6,  "green": 5,  "button": 10, "pir": 17}  # Street 3
]

# Audio (YAMNet & PyAudio)
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK = 4096
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_WINDOW_SECONDS = 0.96
YAMNET_WINDOW_SAMPLES = int(AUDIO_RATE * YAMNET_WINDOW_SECONDS)
EMERGENCY_KEYWORDS = ["siren", "ambulance", "emergency", "police", "fire truck", "alarm"]
EMERGENCY_DETECTION_THRESHOLD = 0.3
MIC_MAPPING_FILE = "mic_street_mapping.json"

# Timing Constants
WAIT_AFTER_BUTTON = 10
MOTION_TIMEOUT = 3
MAX_PEDESTRIAN_TIME = 40
PEDESTRIAN_COOLDOWN_TIME = 90
TRAFFIC_SCAN_INTERVAL = 3  # تم الحفاظ على هذه القيمة
TRAFFIC_RESTORE_DELAY = 5
LONG_WAIT_THRESHOLD = 50
LONG_WAIT_GREEN_DURATION = 30
EMPTY_VERIFICATION_TIME = 5
EMERGENCY_MODE_DURATION = 60

# --- Global State Variables ---
system_lock = threading.Lock()
stop_event = threading.Event()

# Pedestrian Mode
traffic_locked = False
pedestrian_mode_active = False
pedestrian_mode_start_time = 0
last_motion_time = 0
motion_detected_flags = [False] * NUM_STREETS
active_pedestrian_crossing = -1
pedestrian_cooldown_until = [0] * NUM_STREETS

# Emergency Mode
emergency_mode_active = False
emergency_street_index = -1
emergency_mode_start_time = 0
last_emergency_trigger_time = 0

# Traffic Light State
previous_green = None
green_start_time = 0
green_reason = None
green_lock_until = 0
all_streets_empty = False

# Traffic Analysis State
last_process_time = 0
last_wait_update = 0
wait_times = [0] * NUM_STREETS
wait_start_time = [0] * NUM_STREETS
previous_counts = [0] * NUM_STREETS
empty_since = [0] * NUM_STREETS

# Data Queues
audio_data_queue = queue.Queue()

# --- Initialization ---
# Load YOLO model
print("Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO model loaded.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit()

# Load YAMNet model
print("Loading YAMNet model...")
try:
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    yamnet_class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        for row in f:
            yamnet_class_names.append(row.strip().split(",")[2])
    print("✅ YAMNet model loaded.")
except Exception as e:
    print(f"❌ Failed to load YAMNet model: {e}")
    exit()

# Setup cameras
cameras = []
print("Initializing cameras...")
for i, index in enumerate(CAMERA_INDICES):
    try:
        cam = cv2.VideoCapture(index)
        if not cam.isOpened():
            raise Exception(f"Camera {i+1} at {index} failed to open.")
        cameras.append(cam)
        print(f"✅ Camera {i+1} ({index}) initialized.")
    except Exception as e:
        print(f"❌ {e}")
        for c in cameras:
            c.release()
        exit()

# Setup GPIO devices
traffic_signals = []
pedestrian_signals = []
print("Initializing GPIO devices...")
try:
    for i in range(NUM_STREETS):
        traffic_signals.append({
            "red": LED(TRAFFIC_SIGNAL_PINS[i]["red"]),
            "green": LED(TRAFFIC_SIGNAL_PINS[i]["green"])
        })
        pedestrian_signals.append({
            "red": LED(PEDESTRIAN_SIGNAL_PINS[i]["red"]),
            "green": LED(PEDESTRIAN_SIGNAL_PINS[i]["green"]),
            "button": Button(PEDESTRIAN_SIGNAL_PINS[i]["button"], pull_up=False),
            "pir": MotionSensor(PEDESTRIAN_SIGNAL_PINS[i]["pir"], queue_len=1, sample_rate=10, threshold=0.2)
        })
        print(f"✅ GPIO devices for Street {i+1} initialized.")
except Exception as e:
    print(f"❌ Failed to initialize GPIO devices: {e}")
    exit()

# --- Microphone Management Class ---
class MicrophoneManager:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.mic_indices = []
        self.mapping = {}
        self.street_names = [f"شارع {i+1}" for i in range(NUM_STREETS)]
        self.load_mapping()

    def get_available_mics(self):
        available_mics = []
        print("\nSearching for audio input devices...")
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
            print(f"Index {mic['index']}: {mic['name']} (Channels: {mic['channels']})")
        print("="*50)

    def load_mapping(self):
        if os.path.exists(MIC_MAPPING_FILE):
            try:
                with open(MIC_MAPPING_FILE, "r") as f:
                    self.mapping = json.load(f)
                self.mapping = {int(k): v for k, v in self.mapping.items()}
                self.mic_indices = list(self.mapping.keys())
                mapped_street_indices = set(self.mapping.values())
                required_street_indices = set(range(NUM_STREETS))
                if len(self.mic_indices) == NUM_STREETS and mapped_street_indices == required_street_indices:
                    print(f"✅ Loaded mapping from {MIC_MAPPING_FILE}: {self.mapping}")
                    return True
                else:
                    print(f"⚠️ Mapping file '{MIC_MAPPING_FILE}' does not match number of streets ({NUM_STREETS}). Resetting.")
                    self.mapping = {}
                    self.mic_indices = []
            except Exception as e:
                print(f"❌ Error loading mapping file: {e}. Resetting.")
                self.mapping = {}
                self.mic_indices = []
        return False

    def save_mapping(self):
        try:
            with open(MIC_MAPPING_FILE, "w") as f:
                json.dump(self.mapping, f)
            print(f"✅ Saved microphone mapping to {MIC_MAPPING_FILE}")
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
        if len(available_mics) < NUM_STREETS:
            print(f"❌ Cannot setup mapping: Available microphones ({len(available_mics)}) less than number of streets ({NUM_STREETS})!")
            return False

        print(f"\n--- Setting up microphone mapping for {NUM_STREETS} streets ---")
        self.mapping = {}
        self.mic_indices = []
        assigned_indices = set()
        for i in range(NUM_STREETS):
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

mic_manager = MicrophoneManager()

# --- Utility Functions ---
def initialize_all_signals():
    print("Initializing signal states (All Red)...")
    for i in range(NUM_STREETS):
        traffic_signals[i]["red"].on()
        traffic_signals[i]["green"].off()
        pedestrian_signals[i]["red"].on()
        pedestrian_signals[i]["green"].off()
    print("✅ Signals initialized.")

# --- Audio Processing Functions ---
def analyze_audio_with_yamnet(waveform, mic_device_index):
    global emergency_mode_active
    try:
        street_index = mic_manager.get_street_index_for_mic(mic_device_index)
        if street_index == -1:
            print(f"⚠️ Warning: Received audio from unmapped mic index {mic_device_index}")
            return

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32) / 32768.0

        if len(waveform) < YAMNET_WINDOW_SAMPLES:
            waveform = np.pad(waveform, (0, YAMNET_WINDOW_SAMPLES - len(waveform)), "constant")
        elif len(waveform) > YAMNET_WINDOW_SAMPLES:
            waveform = waveform[:YAMNET_WINDOW_SAMPLES]

        scores, embeddings, log_mel_spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()[0]

        detected_emergency = False
        detected_class = "None"
        max_score = 0.0

        for i, score in enumerate(scores_np):
            class_name = yamnet_class_names[i]
            is_emergency_sound = any(keyword in class_name.lower() for keyword in EMERGENCY_KEYWORDS)
            if is_emergency_sound and score > EMERGENCY_DETECTION_THRESHOLD:
                if score > max_score:
                    detected_emergency = True
                    detected_class = class_name
                    max_score = score

        if detected_emergency:
            print(f"🚨 EMERGENCY SOUND DETECTED in Street {street_index + 1} (Mic: {mic_device_index}) - Sound: {detected_class}, Score: {max_score:.2f}")
            trigger_emergency_mode(street_index)

    except Exception as e:
        print(f"❌ Error during YAMNet analysis for mic {mic_device_index}: {e}")

def audio_capture_thread(device_index, data_queue, stop_event):
    p = pyaudio.PyAudio()
    stream = None
    street_index = mic_manager.get_street_index_for_mic(device_index)
    street_name = f"Street {street_index + 1}" if street_index != -1 else f"Unknown Mic {device_index}"
    
    try:
        stream = p.open(format=AUDIO_FORMAT,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        input=True,
                        frames_per_buffer=AUDIO_CHUNK,
                        input_device_index=device_index)
        
        print(f"🎤 Started audio capture thread for {street_name} (Mic Index: {device_index})...")
        
        while not stop_event.is_set():
            try:
                data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                data_queue.put((audio_np, device_index))
                
            except IOError as e:
                if not stop_event.is_set():
                    print(f"❌ Audio IO Error in {street_name} (Mic: {device_index}): {e}")
                    time.sleep(1)
            except Exception as e:
                if not stop_event.is_set():
                    print(f"❌ Unexpected Error in audio capture for {street_name} (Mic: {device_index}): {e}")
                    break

    except Exception as e:
        print(f"❌ Failed to start audio stream for {street_name} (Mic: {device_index}): {e}")
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"❌ Error closing audio stream for {street_name} (Mic: {device_index}): {e}")
        p.terminate()
        print(f"🛑 Stopped audio capture thread for {street_name} (Mic: {device_index}).")

# --- Emergency Mode Functions ---
def trigger_emergency_mode(street_index):
    global emergency_mode_active, emergency_street_index, emergency_mode_start_time, pedestrian_mode_active, last_emergency_trigger_time

    current_time = time.time()
    
    if current_time - last_emergency_trigger_time < EMERGENCY_MODE_DURATION:
        print(f"🚫 Ignoring emergency from Street {street_index + 1} - Another emergency active or cooling down.")
        return

    with system_lock:
        print(f"⚡️ TRIGGERING EMERGENCY MODE for Street {street_index + 1} ⚡️")
        emergency_mode_active = True
        last_emergency_trigger_time = current_time
        emergency_street_index = street_index
        emergency_mode_start_time = current_time

        if pedestrian_mode_active:
            pedestrian_mode_active = False
            active_pedestrian_crossing = -1
            for i in range(NUM_STREETS):
                pedestrian_signals[i]["green"].off()
                pedestrian_signals[i]["red"].on()

        for i in range(NUM_STREETS):
            if i == street_index:
                traffic_signals[i]["red"].off()
                traffic_signals[i]["green"].on()
                print(f"🚦 Emergency: GREEN for Street {i+1}")
            else:
                traffic_signals[i]["green"].off()
                traffic_signals[i]["red"].on()
                print(f"🚦 Emergency: RED for Street {i+1}")

        for i in range(NUM_STREETS):
            pedestrian_signals[i]["green"].off()
            pedestrian_signals[i]["red"].on()

def deactivate_emergency_mode():
    global emergency_mode_active, emergency_street_index
    with system_lock:
        if not emergency_mode_active:
            return
        print(f"✅ Deactivating Emergency Mode for Street {emergency_street_index + 1}.")
        emergency_mode_active = False
        emergency_street_index = -1
        print(f"⏳ Waiting {TRAFFIC_RESTORE_DELAY} seconds before restoring normal traffic logic...")
        time.sleep(TRAFFIC_RESTORE_DELAY)

# --- Pedestrian Mode Functions ---
def activate_pedestrian_mode(crossing_index):
    global traffic_locked, pedestrian_mode_active, pedestrian_mode_start_time, active_pedestrian_crossing
    global wait_times, wait_start_time, previous_counts, empty_since, saved_wait_times, saved_wait_start_time

    saved_wait_times = wait_times.copy()
    saved_wait_start_time = wait_start_time.copy()
    
    current_time = time.time()
    
    with system_lock:
        if emergency_mode_active:
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Emergency Mode is active.")
            return

        if pedestrian_mode_active:
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Already active.")
            return
        
        if current_time < pedestrian_cooldown_until[crossing_index]:
            remaining_cooldown = int(pedestrian_cooldown_until[crossing_index] - current_time)
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Still in cooldown for {remaining_cooldown} seconds.")
            return
        
        print(f"🚶‍♂️ Activating pedestrian mode triggered by crossing {crossing_index + 1}")
        
        for i in range(NUM_STREETS):
            traffic_signals[i]["green"].off()
            traffic_signals[i]["red"].on()
        
        time.sleep(WAIT_AFTER_BUTTON)
        
        for i in range(NUM_STREETS):
            pedestrian_signals[i]["red"].off()
            pedestrian_signals[i]["green"].on()
        
        pedestrian_mode_active = True
        pedestrian_mode_start_time = current_time
        last_motion_time = current_time
        active_pedestrian_crossing = crossing_index
        traffic_locked = True

def deactivate_pedestrian_mode(triggered_by_emergency=False):
    global pedestrian_mode_active, active_pedestrian_crossing, traffic_locked

    with system_lock:
        if not pedestrian_mode_active:
            return
        
        print(f"🛑 Deactivating pedestrian mode.")
        for i in range(NUM_STREETS):
            pedestrian_signals[i]["green"].off()
            pedestrian_signals[i]["red"].on()
        
        pedestrian_mode_active = False
        active_pedestrian_crossing = -1
        traffic_locked = False
        
        if not triggered_by_emergency:
            print(f"⏳ Waiting {TRAFFIC_RESTORE_DELAY} seconds before restoring traffic signals...")

# --- Traffic Analysis and Control Functions ---
def count_vehicles(results):
    count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in VEHICLE_CLASSES:
                count += 1
    return count

def control_traffic_signals():
    global previous_green, green_start_time, green_reason, green_lock_until
    global wait_times, wait_start_time, previous_counts, empty_since, all_streets_empty
    global last_process_time, last_wait_update, traffic_locked
    
    last_traffic_restore_wait = 0
    
    while not stop_event.is_set():
        try:
            current_time = time.time()
            
            if emergency_mode_active:
                if current_time - emergency_mode_start_time >= EMERGENCY_MODE_DURATION:
                    print(f"⏱️ Emergency mode duration ({EMERGENCY_MODE_DURATION} seconds) expired.")
                    deactivate_emergency_mode()
                    last_traffic_restore_wait = current_time
                time.sleep(0.5)
                continue

            if pedestrian_mode_active or traffic_locked:
                time.sleep(0.5)
                continue

            if last_traffic_restore_wait > 0:
                if current_time - last_traffic_restore_wait >= TRAFFIC_RESTORE_DELAY:
                    print("✅ Restore delay finished. Resuming normal traffic control.")
                    last_traffic_restore_wait = 0
                    initialize_all_signals()
                    previous_green = None
                    green_reason = None
                    green_lock_until = 0
                    wait_times = [0] * NUM_STREETS
                    wait_start_time = [0] * NUM_STREETS
                    previous_counts = [0] * NUM_STREETS
                    empty_since = [0] * NUM_STREETS
                    all_streets_empty = False
                    last_process_time = 0
                else:
                    time.sleep(0.5)
                    continue

            if current_time - last_wait_update >= 1:
                for i in range(NUM_STREETS):
                    is_waiting = (previous_green is None or previous_green != i)
                    if is_waiting:
                        if previous_counts[i] > 0 and wait_start_time[i] == 0:
                            wait_start_time[i] = current_time
                            wait_times[i] = 0
                        elif previous_counts[i] > 0 and wait_start_time[i] > 0:
                            wait_times[i] = int(current_time - wait_start_time[i])
                        elif previous_counts[i] == 0:
                            wait_start_time[i] = 0
                            wait_times[i] = 0
                    else:
                        wait_start_time[i] = 0
                        wait_times[i] = 0
                last_wait_update = current_time

            # --- Process cameras and YOLO ---
            if current_time - last_process_time >= TRAFFIC_SCAN_INTERVAL:
                last_process_time = current_time
                frames = []
                valid_reads = True

                for i in range(NUM_STREETS):
                    ret, frame = cameras[i].read()
                    if not ret:
                        print(f"⚠️ Problem reading camera {i+1}")
                        valid_reads = False
                        break
                    frames.append(cv2.resize(frame, (320, 240)))  # تقليل الدقة لتسريع المعالجة

                if not valid_reads:
                    time.sleep(1)
                    continue

                try:
                    results = yolo_model(frames)
                    counts = [count_vehicles([results[i]]) for i in range(NUM_STREETS)]
                except Exception as e:
                    print(f"❌ Error during YOLO processing: {e}")
                    continue

                for i in range(NUM_STREETS):
                    if previous_counts[i] > 0 and counts[i] == 0:
                        if empty_since[i] == 0:
                            empty_since[i] = current_time
                    elif counts[i] > 0:
                        empty_since[i] = 0

                previous_counts = counts.copy()

                if all(count == 0 for count in counts):
                    if not all_streets_empty:
                        print("⚠️ All streets are empty, setting all traffic signals to RED")
                        all_streets_empty = True
                        if previous_green is not None:
                            traffic_signals[previous_green]["green"].off()
                            traffic_signals[previous_green]["red"].on()
                            previous_green = None
                            green_reason = None
                            green_lock_until = 0
                    continue
                else:
                    all_streets_empty = False

                force_change = False
                can_change = green_lock_until <= current_time
                priority_street = -1
                next_green_reason = None

                if previous_green is not None and can_change:
                    if counts[previous_green] == 0 and empty_since[previous_green] > 0 and current_time - empty_since[previous_green] >= EMPTY_VERIFICATION_TIME:
                        print(f"🚫 Street {previous_green+1} confirmed empty, forcing signal change.")
                        force_change = True
                    elif green_reason == "long_wait" and current_time - green_start_time >= LONG_WAIT_GREEN_DURATION:
                        print(f"⏱️ Green duration ended for Street {previous_green+1} (long wait).")
                        force_change = True

                if can_change or force_change:
                    long_wait_streets = [i for i in range(NUM_STREETS) if wait_times[i] >= LONG_WAIT_THRESHOLD and counts[i] > 0]
                    
                    if long_wait_streets:
                        priority_street = max(long_wait_streets, key=lambda i: wait_times[i])
                        next_green_reason = "long_wait"
                        if priority_street != previous_green:
                            force_change = True
                    else:
                        waiting_streets = [i for i in range(NUM_STREETS) if counts[i] > 0 and i != previous_green]
                        if waiting_streets:
                            potential_priority = max(waiting_streets, key=lambda i: counts[i])
                            if previous_green is None or counts[potential_priority] > counts[previous_green] or force_change:
                                priority_street = potential_priority
                                next_green_reason = "high_count"
                                if priority_street != previous_green:
                                    force_change = True
                            else:
                                priority_street = previous_green
                                next_green_reason = green_reason
                        elif previous_green is not None and counts[previous_green] > 0 and not force_change:
                            priority_street = previous_green
                            next_green_reason = green_reason
                        elif force_change:
                            priority_street = -1
                            next_green_reason = None
                        else:
                            priority_street = -1
                            next_green_reason = None

                if (priority_street != previous_green and priority_street != -1 and can_change) or (force_change and priority_street != previous_green):
                    with system_lock:
                        if previous_green is not None:
                            traffic_signals[previous_green]["green"].off()
                            traffic_signals[previous_green]["red"].on()
                            wait_times[previous_green] = 0
                            wait_start_time[previous_green] = 0
                        
                        if priority_street != -1:
                            traffic_signals[priority_street]["red"].off()
                            traffic_signals[priority_street]["green"].on()
                            previous_green = priority_street
                            green_start_time = current_time
                            green_reason = next_green_reason
                            wait_times[priority_street] = 0
                            wait_start_time[priority_street] = 0
                            empty_since[priority_street] = 0
                            
                            if green_reason == "long_wait":
                                green_lock_until = current_time + LONG_WAIT_GREEN_DURATION
                            else:
                                green_lock_until = 0
                        else:
                            previous_green = None
                            green_reason = None
                            green_lock_until = 0

                elif force_change and priority_street == -1 and previous_green is not None:
                    with system_lock:
                        traffic_signals[previous_green]["green"].off()
                        traffic_signals[previous_green]["red"].on()
                        wait_times[previous_green] = 0
                        wait_start_time[previous_green] = 0
                        previous_green = None
                        green_reason = None
                        green_lock_until = 0

            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error in control_traffic_signals: {e}")
            stop_event.set()
            break

# --- Pedestrian Monitoring ---
def monitor_pir_sensor(index):
    pir = pedestrian_signals[index]["pir"]
    
    def motion_detected():
        motion_detected_flags[index] = True
    
    def motion_stopped():
        motion_detected_flags[index] = False
    
    pir.when_motion = motion_detected
    pir.when_no_motion = motion_stopped
    print(f"✅ Started monitoring PIR sensor {index+1}")
    
    stop_event.wait()
    print(f"🛑 Stopped monitoring PIR sensor {index+1}")

def monitor_pedestrian_activity():
    global last_motion_time, pedestrian_mode_active, active_pedestrian_crossing
    
    while not stop_event.is_set():
        if pedestrian_mode_active and active_pedestrian_crossing != -1:
            if emergency_mode_active:
                time.sleep(0.5)
                continue
            
            current_time = time.time()
            elapsed_time = current_time - pedestrian_mode_start_time
            
            if elapsed_time >= MAX_PEDESTRIAN_TIME:
                print(f"⏱️ Pedestrian time expired ({MAX_PEDESTRIAN_TIME} seconds). Stopping pedestrian mode.")
                deactivate_pedestrian_mode()
                continue
            
            motion_at_active_crossing = motion_detected_flags[active_pedestrian_crossing]
            
            if motion_at_active_crossing:
                last_motion_time = current_time
            elif current_time - last_motion_time >= MOTION_TIMEOUT:
                print(f"🛑 No pedestrian motion for {MOTION_TIMEOUT} seconds at crossing {active_pedestrian_crossing+1}. Stopping pedestrian mode.")
                deactivate_pedestrian_mode()
        
        time.sleep(0.5)

def handle_pedestrian_button(index):
    button = pedestrian_signals[index]["button"]
    
    while not stop_event.is_set():
        try:
            button.wait_for_press(timeout=None)
            
            if stop_event.is_set():
                break
            
            print(f"🔘 [Pedestrian Signal {index + 1}] Button pressed, waiting {WAIT_AFTER_BUTTON} seconds...")
            
            wait_end_time = time.time() + WAIT_AFTER_BUTTON
            while time.time() < wait_end_time and not stop_event.is_set():
                time.sleep(0.1)
            
            if stop_event.is_set():
                break
            
            activate_pedestrian_mode(index)
            
            while pedestrian_mode_active and not emergency_mode_active and not stop_event.is_set():
                time.sleep(0.5)
            
            if stop_event.is_set():
                break
            
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"❌ Error in handle_pedestrian_button {index+1}: {e}")
            time.sleep(1)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🚦 Starting Integrated Smart Traffic Light System...")
    
    initialize_all_signals()
    
    print("\nLoading microphone mapping...")
    if not mic_manager.load_mapping():
        print(f"❌ Error: Microphone mapping file '{MIC_MAPPING_FILE}' not found or invalid.")
        print("Please ensure the mapping file exists alongside the script.")
        exit(1)

    threads = []

    print("\nStarting Audio Capture Threads...")
    for mic_index in mic_manager.get_mic_indices():
        thread = threading.Thread(target=audio_capture_thread, args=(mic_index, audio_data_queue, stop_event), daemon=True)
        threads.append(thread)
        thread.start()

    print("\nStarting Pedestrian Button Threads...")
    for i in range(NUM_STREETS):
        thread = threading.Thread(target=handle_pedestrian_button, args=(i,), daemon=True)
        threads.append(thread)
        thread.start()

    print("\nStarting PIR Sensor Threads...")
    for i in range(NUM_STREETS):
        thread = threading.Thread(target=monitor_pir_sensor, args=(i,), daemon=True)
        threads.append(thread)
        thread.start()

    print("\nStarting Pedestrian Activity Monitoring Thread...")
    pedestrian_activity_thread = threading.Thread(target=monitor_pedestrian_activity, daemon=True)
    threads.append(pedestrian_activity_thread)
    pedestrian_activity_thread.start()

    print("\nStarting Main Control Loop...")
    main_control_thread = threading.Thread(target=control_traffic_signals)
    threads.append(main_control_thread)
    main_control_thread.start()

    print("\nStarting Audio Processing Loop...")
    try:
        while not stop_event.is_set():
            try:
                audio_np, mic_device_index = audio_data_queue.get(block=True, timeout=0.5)
                analyze_audio_with_yamnet(audio_np, mic_device_index)
                audio_data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in audio processing loop: {e}")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard Interrupt detected. Stopping system...")
    except Exception as e:
        print(f"❌ Critical error in main loop: {e}. Stopping system...")
    finally:
        print("\nInitiating shutdown sequence...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5.0)
        print("🧹 Cleaning up resources...")
        mic_manager.terminate()
        for cam in cameras:
            cam.release()
        cv2.destroyAllWindows()
        print("🛑 System stopped.")
