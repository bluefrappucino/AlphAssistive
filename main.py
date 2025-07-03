# --- Import Libraries ---
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import pyttsx3
from dynamixel_sdk import *
import mediapipe as mp
import speech_recognition as sr
import threading
import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv
import pandas as pd

# =============================================================================
# --- KONFIGURASI DAN INISIALISASI GLOBAL ---
# =============================================================================

# --- Konfigurasi Dynamixel ---
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
DXL_ID_HORIZONTAL = 3
DXL_ID_VERTICAL = 1
BAUDRATE = 57600
DEVICENAME = 'COM3'

TORQUE_ENABLE = 1
LIMITS_X = (1461, 2499)
LIMITS_Y = (2199, 2716)
Kp = 0.7
threshold = 5

# --- Konfigurasi Analisis & Logging ---
OUTPUT_DIR = r"D:\record for progress\hasil_analisis"

# --- Variabel Global untuk Pengumpulan Data ---
hand_position_history = []
servo_position_history = []
tracking_error_history = [] 
feedback_counts = {}
start_time_hand_detection = None

# --- Inisialisasi Hardware ---
portHandler = None
packetHandler = None
pipeline = None

def initialize_hardware():
    """Menginisialisasi Dynamixel dan Kamera RealSense."""
    global portHandler, packetHandler, pipeline
    
    try:
        portHandler = PortHandler(DEVICENAME)
        packetHandler = PacketHandler(PROTOCOL_VERSION)
        portHandler.openPort()
        portHandler.setBaudRate(BAUDRATE)
        packetHandler.write1ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write1ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        print("[INFO] Dynamixel berhasil diinisialisasi.")
    except Exception as e:
        print(f"[ERROR] Gagal menginisialisasi Dynamixel di {DEVICENAME}. Error: {e}")
        exit()

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        print("[INFO] Kamera RealSense berhasil diinisialisasi.")
    except Exception as e:
        print(f"[ERROR] Gagal menginisialisasi kamera. Error: {e}")
        if portHandler:
            portHandler.closePort()
        exit()

def cleanup():
    """Mematikan semua koneksi dan jendela."""
    global pipeline, portHandler
    if pipeline:
        pipeline.stop()
    if portHandler:
        portHandler.closePort()
    cv2.destroyAllWindows()
    print("[INFO] Sesi selesai. Semua koneksi ditutup.")

def evaluate_segmentation_error(target_color, duration=2):
    print(f"[EVAL] Mengevaluasi error segmentasi warna '{target_color}' selama {duration} detik...")
    if pipeline is None:
        print("[ERROR] Kamera belum diinisialisasi. Panggil initialize_hardware() dulu.")
        return
    
    start = time.time()
    total_frames = 0
    frames_with_contour = 0

    while time.time() - start < duration:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = get_color_mask(hsv, target_color)

        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                frames_with_contour += 1
        total_frames += 1

    error_rate = 1.0 - (frames_with_contour / total_frames if total_frames > 0 else 0)
    print(f"[RESULT] Akurasi Segmentasi: {100 * (1 - error_rate):.2f}% ({frames_with_contour}/{total_frames} frame)")
    print(f"[RESULT] Error Rate: {100 * error_rate:.2f}%")

# =============================================================================
# --- FUNGSI-FUNGSI PEMBANTU ---
# =============================================================================

def speak_async(text):
    """Menjalankan text-to-speech di thread terpisah agar tidak memblokir."""
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] Gagal mengeluarkan suara: {e}")
    threading.Thread(target=run, daemon=True).start()

def apply_limits(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def is_hand_closed(hand_landmarks, mp_hands_instance):
    """Mendeteksi genggaman berdasarkan posisi Y"""
    if not hand_landmarks:
        return False
    closed_count = 0
    fingers = [
        (mp_hands_instance.HandLandmark.INDEX_FINGER_TIP, mp_hands_instance.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands_instance.HandLandmark.MIDDLE_FINGER_TIP, mp_hands_instance.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands_instance.HandLandmark.RING_FINGER_TIP, mp_hands_instance.HandLandmark.RING_FINGER_MCP),
        (mp_hands_instance.HandLandmark.PINKY_TIP, mp_hands_instance.HandLandmark.PINKY_MCP)
    ]
    for tip, mcp in fingers:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
            closed_count += 1
    return closed_count >= 3

def get_color_mask(hsv_image, color_name):
    """Membuat mask berdasarkan warna"""
    if color_name == "red":
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    elif color_name == "green":
        lower = np.array([42, 43, 108])
        upper = np.array([84, 255, 255])
        return cv2.inRange(hsv_image, lower, upper)
    elif color_name == "blue":
        lower = np.array([42, 45, 160])
        upper = np.array([100, 71, 255])
        return cv2.inRange(hsv_image, lower, upper)
    elif color_name == "yellow":
        lower = np.array([18, 54, 191])
        upper = np.array([83, 137, 255])
        return cv2.inRange(hsv_image, lower, upper)
    elif color_name == "orange":
        lower = np.array([0, 113, 154])
        upper = np.array([60, 215, 255])
        return cv2.inRange(hsv_image, lower, upper)
    return None

# =============================================================================
# --- TAHAP 1: INPUT SUARA ---
# =============================================================================

def get_target_color_from_voice():
    r = sr.Recognizer()
    mic = sr.Microphone()
    color_options = ["red", "green", "blue", "yellow", "orange"]
    target_color = None
    print("[INFO] Menunggu perintah suara: '<color> object'")
    speak_async("Please tell me which object you want to pick up.")
    while target_color is None:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            print("[INFO] Mendengarkan perintah warna...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                speak_async("I didn't hear anything. Please try again.")
                continue
        try:
            command = r.recognize_google(audio, language='en-US').lower()
            print(f"[INFO] Perintah diterima: '{command}'")
            detected_color = next((color for color in color_options if color in command), None)
            if detected_color:
                target_color = detected_color
                return target_color
            else:
                speak_async("I didn't recognize the color. Please say it again clearly.")
        except sr.UnknownValueError:
            speak_async("I could not understand. Please repeat.")
        except sr.RequestError:
            speak_async("Sorry, I'm having trouble with my speech service.")
            return None
    return None

# =============================================================================
# --- TAHAP 2: SCANNING & TRACKING OBJEK ---
# =============================================================================

def scan_for_object(target_color):
    scan_sequence = [
        ("kanan",  (2465, 1817)),
        ("normal", (2465, 2048)),
        ("kiri",   (2465, 2455)),
        ("normal", (2465, 2048)),
        ("atas",   (2608, 2048)),
        ("normal", (2465, 2048)),
        ("bawah",  (2223, 2048)),
        ("normal", (2465, 2048)),
    ]

    def wait_until_position_reached(target_y, target_x, tolerance=10, timeout=2):
        start_time = time.time()
        while time.time() - start_time < timeout:
            curr_x = packetHandler.read4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_PRESENT_POSITION)[0]
            curr_y = packetHandler.read4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_PRESENT_POSITION)[0]
            if abs(curr_x - target_x) < tolerance and abs(curr_y - target_y) < tolerance:
                return
            time.sleep(0.05)

    for direction, (pos_y, pos_x) in scan_sequence:
        print(f"[SCAN] Menggerakkan kamera ke arah {direction.upper()}")
        packetHandler.write4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_GOAL_POSITION, pos_y)
        packetHandler.write4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_GOAL_POSITION, pos_x)
        wait_until_position_reached(pos_y, pos_x)

        for _ in range(5):  # buang frame awal
            pipeline.wait_for_frames()

        start_time = time.time()
        valid_detections = 0
        best_coords = None

        while time.time() - start_time < 2:  # deteksi selama 2 detik
            frames = pipeline.wait_for_frames()
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = get_color_mask(hsv, target_color)
            if mask is None:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) < 300:
                    continue

                x, y, w, h = cv2.boundingRect(largest)
                obj_x, obj_y = x + w // 2, y + h // 2
                depth_roi = np.asanyarray(depth_frame.get_data())[y:y+h, x:x+w]
                valid_depth = depth_roi[(depth_roi > 100) & (depth_roi < 2000)]
                obj_z = int(np.median(valid_depth)) if valid_depth.size > 0 else 0

                if obj_z > 0:
                    valid_detections += 1
                    best_coords = (obj_x, obj_y, obj_z)

                    # Tampil jendela opencv
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(color_image, f"Obj Detected ({obj_x},{obj_y},{obj_z}mm)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tampil video setiap frame
            cv2.imshow("Scanning Object", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if valid_detections >= 5:
                print(f"[SCAN] Objek ditemukan di arah {direction.upper()} pada ({best_coords[0]}, {best_coords[1]}, {best_coords[2]}mm)")
                cv2.destroyWindow("Scanning Object")
                return best_coords

    speak_async("Object not found")
    print("[SCAN] Menyimpan snapshot dan mask terakhir untuk dokumentasi skripsi...")

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[ERROR] Tidak ada frame terakhir untuk disimpan.")
        else:
            color_image = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = get_color_mask(hsv, target_color)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fail_dir = os.path.join(OUTPUT_DIR, "gagal_deteksi")
            os.makedirs(fail_dir, exist_ok=True)

            rgb_path = os.path.join(fail_dir, f"{target_color}_{timestamp}_rgb.png")
            mask_path = os.path.join(fail_dir, f"{target_color}_{timestamp}_mask.png")

            cv2.imwrite(rgb_path, color_image)
            if mask is not None:
                cv2.imwrite(mask_path, mask)
                print(f"[SCAN] Gambar & mask disimpan:\n - {rgb_path}\n - {mask_path}")
            else:
                print(f"[WARNING] Mask tidak ditemukan untuk warna: {target_color}")
                print(f"[SCAN] Gambar RGB tetap disimpan: {rgb_path}")

    except Exception as e:
        print(f"[ERROR] Gagal menyimpan snapshot gagal: {e}")


    cv2.destroyWindow("Scanning Object")
    speak_async("Object not found")
    return None, None, None

def track_object(target_color, initial_obj_x, initial_obj_y, initial_obj_z):
    global servo_position_history, tracking_error_history
    obj_x, obj_y, obj_z = initial_obj_x, initial_obj_y, initial_obj_z
    stable_counter = 0
    stable_required = 10 # Dibuat lebih tinggi agar stabil
    
    print(f"[INFO] Melanjutkan tracking dari posisi: ({obj_x}, {obj_y}, {obj_z})")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if not color_frame or not depth_frame: continue

        dxl_pos_x = packetHandler.read4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_PRESENT_POSITION)[0]
        dxl_pos_y = packetHandler.read4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_PRESENT_POSITION)[0]
        
        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = get_color_mask(hsv, target_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            obj_x, obj_y = x + w // 2, y + h // 2
            
            depth_roi = np.asanyarray(depth_frame.get_data())[y:y+h, x:x+w]
            valid_depth = depth_roi[(depth_roi > 100) & (depth_roi < 2000)]
            if valid_depth.size > 0: obj_z = int(np.median(valid_depth))

            frame_center_x, frame_center_y = color_image.shape[1] // 2, color_image.shape[0] // 2
            error_x = frame_center_x - obj_x
            error_y = frame_center_y - obj_y
            
            current_time = time.time() # Gunakan waktu absolut untuk plotting
            servo_position_history.append((current_time, dxl_pos_x, dxl_pos_y))
            tracking_error_history.append((current_time, error_x, error_y))

            if abs(error_x) > threshold:
                target_pos_x = apply_limits(int(dxl_pos_x + Kp * error_x), *LIMITS_X)
                packetHandler.write4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_GOAL_POSITION, target_pos_x)
            if abs(error_y) > threshold:
                target_pos_y = apply_limits(int(dxl_pos_y + Kp * error_y), *LIMITS_Y)
                packetHandler.write4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_GOAL_POSITION, target_pos_y)
            
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(color_image, f"Cube ({obj_x},{obj_y},{int(obj_z/10)}cm)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if abs(error_x) < 10 and abs(error_y) < 10:
                stable_counter += 1
                if stable_counter >= stable_required:
                    print("[INFO] Objek dikunci. Siap ke deteksi tangan.")
                    speak_async("Okay, You can start to grab it!")
                    cv2.destroyWindow("Tracking Objek")
                    return obj_x, obj_y, obj_z
            else:
                stable_counter = 0
        else:
            stable_counter = 0

        cv2.imshow("Tracking Objek", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, None, None

# =============================================================================
# --- TAHAP 3: DETEKSI TANGAN & FEEDBACK ---
# =============================================================================

def detect_hand_and_guide(obj_coords):
    global hand_position_history, feedback_counts, start_time_hand_detection
    obj_x, obj_y, obj_z = obj_coords
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    spatial, temporal = rs.spatial_filter(), rs.temporal_filter()
    holding_object, last_feedback, last_talk_time, last_hand_pos = False, "", 0, None
    start_time_hand_detection = time.time()
    
    while not holding_object:
        frames = pipeline.wait_for_frames()
        depth_frame_filtered = spatial.process(temporal.process(frames.get_depth_frame()))
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame_filtered: continue
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame_filtered.get_data())
        frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        feedback = ""
        
        is_grasped_now = False # Default
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            palm_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_cx, palm_cy = int(palm_mcp.x * 640), int(palm_mcp.y * 480)
            
            hand_z_list = []
            for idx in [0, 5, 9, 13, 17]:
                lm = hand_landmarks.landmark[idx]
                x, y = int(lm.x * 640), int(lm.y * 480)
                if 0 <= x < 640 and 0 <= y < 480:
                    z = depth_image[y, x]
                    if 100 < z < 2000: hand_z_list.append(z)
            hand_z = int(np.median(hand_z_list)) if hand_z_list else 0
            last_hand_pos = (palm_cx, palm_cy, hand_z)
            
            # --- LOGIKA XYZ ---
            if obj_x - 30 <= palm_cx < obj_x - 10: feedback += "A little bit right. "
            elif palm_cx < obj_x - 30: feedback += "Move right. "
            elif obj_x + 10 < palm_cx <= obj_x + 30: feedback += "A little bit left. "
            elif palm_cx > obj_x + 30: feedback += "Move left. "

            if obj_y - 30 <= palm_cy < obj_y - 10: feedback += "A little bit down. "
            elif palm_cy < obj_y - 30: feedback += "Move down. "
            elif obj_y + 10 < palm_cy <= obj_y + 30: feedback += "A little bit up. "
            elif palm_cy > obj_y + 30: feedback += "Move up. "
            
            if hand_z > 0:
                selisih_z = hand_z - obj_z
                toleransi_z = max(50, int(obj_z * 0.15))
                if toleransi_z < selisih_z <= toleransi_z + 100: feedback += "A little bit backward. "
                elif selisih_z > toleransi_z + 100: feedback += "Move backward. "
                elif -toleransi_z - 100 <= selisih_z < -toleransi_z: feedback += "A little bit forward. "
                elif selisih_z < -toleransi_z - 100: feedback += "Move forward. "

            # --- AKHIR LOGIKA XYZ ---
            
            is_grasped_now = is_hand_closed(hand_landmarks, mp_hands)

            if not feedback and not holding_object:
                feedback = "Grab the object now!"
            elif not feedback and holding_object:
                feedback = "Object successfully grasped."
                speak_async("Object successfully grasped.")

            if not holding_object and is_grasped_now:
                holding_object = True
                print("[INFO] Hand is closed. Object successfully grasped.")
                feedback = "Object successfully grasped"
                speak_async("Object successfully grasped.")
            
            if feedback and (feedback != last_feedback or (time.time() - last_talk_time) > 2):
                speak_async(feedback)
                feedback_counts[feedback.strip()] = feedback_counts.get(feedback.strip(), 0) + 1
                last_feedback, last_talk_time = feedback, time.time()
                print("[INFO]", feedback)

            # Logging data untuk analisis
            if hand_z > 0:
                current_time = time.time() - start_time_hand_detection
                hand_state = "closed" if is_grasped_now else "open"
                hand_position_history.append((current_time, obj_x, obj_y, obj_z, palm_cx, palm_cy, hand_z,
                                              obj_x - palm_cx, obj_y - palm_cy, obj_z - hand_z,
                                              hand_state, feedback.strip(), feedback_counts.get(feedback.strip(), 0)))

            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(color_image, f"Target Obj: ({obj_x},{obj_y},{int(obj_z/10)}cm)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if last_hand_pos:
            cv2.putText(color_image, f"Hand Pos: ({last_hand_pos[0]},{last_hand_pos[1]},{int(last_hand_pos[2]/10)}cm)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if holding_object:
            cv2.putText(color_image, "GRASPED!", (180, 240), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 0), 3)
        
        cv2.imshow("Deteksi Tangan", color_image)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (holding_object and feedback == "Object successfully grasped"):
            if holding_object:
                speak_async("Object successfully grasped")
                time.sleep(2)
            break
            
    cv2.destroyWindow("Deteksi Tangan")
    return holding_object

# =============================================================================
# --- TAHAP 4: ANALISIS & VISUALISASI DATA ---
# =============================================================================

def plot_professional_visual_servoing(error_history, servo_history):
    """4 plot untuk analisis visual servoing."""
    if not error_history or not servo_history or len(servo_history) < 2:
        print("[ANALISIS] Tidak cukup data untuk plot visual servoing.")
        return
        
    print("[VISUAL] Grafik analisis Visual Servoing...")
    
    # Normalisasi waktu agar dimulai dari 0
    start_time = error_history[0][0]
    error_history = [(t - start_time, ex, ey) for t, ex, ey in error_history]
    servo_history = [(t - start_time, px, py) for t, px, py in servo_history]
    
    err_df = pd.DataFrame(error_history, columns=["time", "error_x", "error_y"])
    servo_df = pd.DataFrame(servo_history, columns=["time", "pos_x", "pos_y"])

    servo_df['vel_x'] = servo_df['pos_x'].diff() / servo_df['time'].diff()
    servo_df['vel_y'] = servo_df['pos_y'].diff() / servo_df['time'].diff()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analisis Kinerja Visual Servoing', fontsize=16)

    axs[0, 0].plot(err_df["time"], err_df["error_x"], label="Error X", color='r')
    axs[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[0, 0].set_title('Error Sumbu X terhadap Waktu'); axs[0, 0].set_xlabel('Waktu (s)'); axs[0, 0].set_ylabel('Error (pixels)')
    axs[0, 0].grid(True, linestyle=':'); axs[0, 0].legend()

    axs[0, 1].plot(err_df["time"], err_df["error_y"], label="Error Y", color='b')
    axs[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[0, 1].set_title('Error Sumbu Y terhadap Waktu'); axs[0, 1].set_xlabel('Waktu (s)'); axs[0, 1].set_ylabel('Error (pixels)')
    axs[0, 1].grid(True, linestyle=':'); axs[0, 1].legend()

    axs[1, 0].plot(err_df["error_x"], err_df["error_y"], label="Jalur Error", color='purple', marker='.', markersize=3, linestyle='-')
    axs[1, 0].scatter(err_df["error_x"].iloc[0], err_df["error_y"].iloc[0], marker='o', color='g', s=100, label='Mulai')
    axs[1, 0].scatter(err_df["error_x"].iloc[-1], err_df["error_y"].iloc[-1], marker='X', color='r', s=100, label='Selesai')
    axs[1, 0].invert_xaxis() #dibuat inverse agar mudah dilihat
    axs[1, 0].set_title('Jalur Error (Phase Portrait)'); axs[1, 0].set_xlabel('Error X (pixels)'); axs[1, 0].set_ylabel('Error Y (pixels)')
    axs[1, 0].axvline(0, color='black', ls='--', lw=1); axs[1, 0].axhline(0, color='black', ls='--', lw=1)
    axs[1, 0].grid(True, linestyle=':'); axs[1, 0].legend(); axs[1, 0].axis('equal')

    axs[1, 1].plot(servo_df["time"], servo_df["vel_x"].fillna(0), label="Kecepatan Servo X", color='orange', alpha=0.8)
    axs[1, 1].plot(servo_df["time"], servo_df["vel_y"].fillna(0), label="Kecepatan Servo Y", color='cyan', alpha=0.8)
    axs[1, 1].set_title('Upaya Kontrol (Kecepatan Servo) terhadap Waktu'); axs[1, 1].set_xlabel('Waktu (s)'); axs[1, 1].set_ylabel('Kecepatan (unit/s)')
    axs[1, 1].grid(True, linestyle=':'); axs[1, 1].legend()

    plt.tight_layout(pad=3.0)
    plt.show()

def save_and_visualize_results(obj_coords, target_color):
    """Menyimpan data, membuat plot, dan merekap hasil eksperimen."""
    print("[ANALYSIS] Memulai proses analisis dan visualisasi...")
    save_data_to_csv(obj_coords, target_color)
    plot_feedback_counts()
    plot_hand_movement(obj_coords)
    plot_professional_visual_servoing(tracking_error_history, servo_position_history)

def save_data_to_csv(obj_coords, target_color):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"analisis_{target_color}_{timestamp}.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        total_turn = feedback_counts.get("Move left.", 0) + feedback_counts.get("Move right.", 0)
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow(["Target Color", target_color]); writer.writerow(["Target Object Coords (px, px, mm)", f"{obj_coords[0]},{obj_coords[1]},{obj_coords[2]}"]); writer.writerow(["Total Turn Steps", total_turn])
        writer.writerow([]); writer.writerow(["--- VOICE FEEDBACK COUNT ---"])
        for command, count in feedback_counts.items(): writer.writerow([command, count])
        writer.writerow([]); writer.writerow(["--- HAND POSITION TIME SERIES ---"])
        writer.writerow(["time_s", "obj_x", "obj_y", "obj_z_mm", "hand_x", "hand_y", "hand_z_mm", "error_x", "error_y", "error_z", "hand_state", "voice_command", "command_count"])
        writer.writerows(hand_position_history)
        writer.writerow([]); writer.writerow(["--- SERVO POSITION TIME SERIES (time_s, posX_raw, posY_raw) ---"])
        writer.writerows(servo_position_history)
        writer.writerow([]); writer.writerow(["--- TRACKING ERROR TIME SERIES (time_s, error_x_px, error_y_px) ---"])
        writer.writerows(tracking_error_history)
    print(f"[ANALYSIS] Data berhasil disimpan di: {filename}")

def plot_feedback_counts():
    if not feedback_counts: return
    total_turn_steps = feedback_counts.get("Move left.", 0) + feedback_counts.get("Move right.", 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.barh(list(feedback_counts.keys()), list(feedback_counts.values()), color='skyblue')
    ax1.set_xlabel("Jumlah Feedback"); ax1.set_title("Jumlah Semua Feedback Suara")
    ax2.bar(['Total Turn Steps'], [total_turn_steps], color='salmon')
    ax2.set_ylabel("Jumlah"); ax2.set_title("Jumlah Langkah 'Turn' (Kiri + Kanan)")
    ax2.set_ylim(0, max(total_turn_steps + 2, 10))
    plt.tight_layout(); plt.show()

def plot_hand_movement(obj_coords):
    if not hand_position_history: return
    times, xs, ys, zs = zip(*[(r[0], r[4], r[5], r[6]) for r in hand_position_history])
    obj_x, obj_y, obj_z = obj_coords
    perfect_x_lim, perfect_y_lim = (obj_x-10, obj_x+10), (obj_y-10, obj_y+10)
    perfect_z_lim = (obj_z - max(50, int(obj_z*0.15)), obj_z + max(50, int(obj_z*0.15)))
    tolerance_x_lim, tolerance_y_lim = (obj_x-30, obj_x+30), (obj_y-30, obj_y+30)
    tolerance_z_lim = (obj_z - (max(50, int(obj_z*0.15)) + 100), obj_z + (max(50, int(obj_z*0.15)) + 100))
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(xs, ys, zs, label='Hand Path', color='c', marker='.', markersize=3, alpha=0.7)
    if xs:
        ax1.scatter(xs[0], ys[0], zs[0], color='orange', s=100, label='Start', depthshade=False)
        ax1.scatter(xs[-1], ys[-1], zs[-1], color='blue', s=120, label='Grasp', depthshade=False)
    ax1.scatter(obj_x, obj_y, obj_z, color='red', s=150, marker='X', label='Target', depthshade=False)
    def draw_box(ax, x_lim, y_lim, z_lim, color, alpha):
        x, y, z = np.array(x_lim), np.array(y_lim), np.array(z_lim)
        pts=np.array([[x[0],y[0],z[0]],[x[0],y[1],z[0]],[x[1],y[1],z[0]],[x[1],y[0],z[0]],[x[0],y[0],z[1]],[x[0],y[1],z[1]],[x[1],y[1],z[1]],[x[1],y[0],z[1]]])
        verts=[[pts[0],pts[1],pts[2],pts[3]],[pts[4],pts[5],pts[6],pts[7]],[pts[0],pts[1],pts[5],pts[4]],[pts[2],pts[3],pts[7],pts[6]],[pts[1],pts[2],pts[6],pts[5]],[pts[4],pts[7],pts[3],pts[0]]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, linewidths=0.5, alpha=alpha))
    draw_box(ax1, tolerance_x_lim, tolerance_y_lim, tolerance_z_lim, 'gray', 0.1)
    draw_box(ax1, perfect_x_lim, perfect_y_lim, perfect_z_lim, 'green', 0.2)
    ax1.set_title('3D Hand Path'); ax1.set_xlabel('X (px)'); ax1.set_ylabel('Y (px)'); ax1.set_zlabel('Z (mm)')
    ax1.invert_yaxis(); ax1.legend()
    axes = [fig.add_subplot(3, 2, 2), fig.add_subplot(3, 2, 4), fig.add_subplot(3, 2, 6)]
    data, targets = [xs, ys, zs], [obj_x, obj_y, obj_z]
    tols, perfs = [tolerance_x_lim, tolerance_y_lim, tolerance_z_lim], [perfect_x_lim, perfect_y_lim, perfect_z_lim]
    labels, units = ['X', 'Y', 'Z'], ['px', 'px', 'mm']
    for i, ax in enumerate(axes):
        ax.plot(times, data[i], label=f'Hand {labels[i]}', color='c')
        ax.axhline(y=targets[i], color='r', ls='--', label=f'Target {labels[i]}')
        ax.axhspan(*tols[i], alpha=0.1, color='gray')
        ax.axhspan(*perfs[i], alpha=0.2, color='green')
        ax.set_title(f'{labels[i]} vs Time'); ax.set_ylabel(f'{labels[i]} ({units[i]})'); ax.legend(fontsize='small'); ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(pad=2.0); plt.show()

# =============================================================================
# --- FUNGSI UTAMA (MAIN EXECUTION) ---
# =============================================================================

def main():
    initialize_hardware()
    target_color = get_target_color_from_voice()
    if not target_color:
        evaluate_segmentation_error("red", duration=2)
        cleanup(); return
    
    print("[INFO] Melakukan scanning awal...")
    obj_coords = scan_for_object(target_color)
    if not all(obj_coords):
        cleanup(); return
    
    final_obj_coords = track_object(target_color, *obj_coords)
    if not all(final_obj_coords):
        cleanup(); return
        
    grasp_successful = detect_hand_and_guide(final_obj_coords)
    if grasp_successful:
        save_and_visualize_results(final_obj_coords, target_color)
    else:
        print("[INFO] Proses tidak selesai, analisis tidak dijalankan.")
        
    cleanup()

if __name__ == '__main__':
    main()