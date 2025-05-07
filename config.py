# -*- coding: utf-8 -*-

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
AUDIO_FORMAT = 1 # pyaudio.paInt16 (using 1 as placeholder, will be resolved by pyaudio)
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000  # Required by YAMNet
AUDIO_CHUNK = 4096  # Audio buffer size
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_WINDOW_SECONDS = 0.96
YAMNET_WINDOW_SAMPLES = int(AUDIO_RATE * YAMNET_WINDOW_SECONDS)
EMERGENCY_KEYWORDS = ["siren", "ambulance", "emergency", "police", "fire truck", "alarm"]
EMERGENCY_DETECTION_THRESHOLD = 0.3
MIC_MAPPING_FILE = "mic_street_mapping.json"

# Timing Constants
WAIT_AFTER_BUTTON = 5             # Delay before activating pedestrian mode
MOTION_TIMEOUT = 3                # No motion time to stop pedestrian green light
MAX_PEDESTRIAN_TIME = 40         # Maximum time for pedestrian green light
PEDESTRIAN_COOLDOWN_TIME = 60    # Cooldown before allowing next pedestrian cycle per crossing
TRAFFIC_SCAN_INTERVAL = 3        # Interval for checking traffic via cameras
TRAFFIC_RESTORE_DELAY = 5        # Wait time before restoring traffic signals after pedestrian/emergency mode
LONG_WAIT_THRESHOLD = 50         # Long wait threshold for cars (seconds)
LONG_WAIT_GREEN_DURATION = 30    # Green light duration for long wait state
EMPTY_VERIFICATION_TIME = 5      # Time to verify street emptiness before changing green light
EMERGENCY_MODE_DURATION = 60     # Duration for emergency mode green light (seconds)

