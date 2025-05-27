import cv2
import numpy as np
import pyrealsense2 as rs
import time
import pyttsx3
from dynamixel_sdk import *
import mediapipe as mp
import speech_recognition as sr
import threading

# --------------------------- #
# KONFIGURASI DYNAMIXEL      #
# --------------------------- #
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
DXL_ID_HORIZONTAL = 3
DXL_ID_VERTICAL = 1
BAUDRATE = 57600
DEVICENAME = 'COM40'

TORQUE_ENABLE = 1
LIMITS_X = (1461, 2693)
LIMITS_Y = (1883, 2785)
Kp = 0.7
threshold = 5

# --------------------------- #
# INISIALISASI DYNAMIXEL     #
# --------------------------- #
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# --------------------------- #
# INISIALISASI KAMERA        #
# --------------------------- #
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# --------------------------- #
# INPUT SUARA SEBELUM MULAI  #
# --------------------------- #
r = sr.Recognizer()
mic = sr.Microphone()
print("[INFO] Waiting for voice command: 'I want to pick up the <color> object'")

target_color = None

while True:
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("[INFO] Listening...")
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio, language='en-US').lower()
        print(f"[INFO] Command heard: {command}")
        if "red" in command:
            target_color = "red"
            print("[INFO] Command recognized. Starting to track red object...")
            break
        elif "green" in command:
            target_color = "green"
            print("[INFO] Command recognized. Starting to track green object...")
            break
        elif "blue" in command:
            target_color = "blue"
            print("[INFO] Command recognized. Starting to track blue object...")
            break
        elif "yellow" in command:
            target_color = "yellow"
            print("[INFO] Command recognized. Starting to track yellow object...")
            break
        elif "orange" in command:
            target_color = "orange"
            print("[INFO] Command recognized. Starting to track orange object...")
            break
        else:
            print("[INFO] Color not recognized. Please try again.")
    except sr.UnknownValueError:
        print("[INFO] Could not understand audio.")
    except sr.RequestError:
        print("[ERROR] Could not access the speech recognition service.")

# --------------------------- #
# FUNGSI PEMBANTU            #
# --------------------------- #
def apply_limits(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def is_hand_closed(hand_landmarks):
    closed_count = 0
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]
    for tip, mcp in fingers:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
            closed_count += 1
    return closed_count >= 3


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()

def detect_target_in_frame(color_image, hsv, target_color):
    if target_color == "red":
        lower_red1 = np.array([46, 22, 131])
        upper_red1 = np.array([10, 81, 255])
        lower_red2 = np.array([160, 101, 92])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif target_color == "green":
        lower = np.array([45, 78, 66])
        upper = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif target_color == "blue":
        lower = np.array([82, 32, 147])
        upper = np.array([180, 92, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif target_color == "yellow":
        lower = np.array([13, 0, 157])
        upper = np.array([71, 135, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif target_color == "orange":
        lower = np.array([7, 95, 154])
        upper = np.array([26, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

# def scan_for_object():
#     scan_positions = {
#         "kanan": 1817,
#         "kiri": 2693,
#         "atas": 2785,
#         "bawah": 1883
#     }

#     for direction, pos in scan_positions.items():
#         print(f"[SCAN] Menggerakkan kamera ke arah {direction.upper()}")
#         if direction in ["kanan", "kiri"]:
#             packetHandler.write4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_GOAL_POSITION, pos)
#         else:
#             packetHandler.write4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_GOAL_POSITION, pos)
#         time.sleep(1.5)

#         for _ in range(5):
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if not color_frame or not depth_frame:
#                 continue

#             color_image = np.asanyarray(color_frame.get_data())
#             depth_image = np.asanyarray(depth_frame.get_data())
#             hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

#             # Gunakan kembali deteksi berdasarkan warna target
#             if target_color == "red":
#                 lower_red1 = np.array([46, 22, 131])
#                 upper_red1 = np.array([10, 81, 255])
#                 lower_red2 = np.array([160, 101, 92])
#                 upper_red2 = np.array([180, 255, 255])
#                 mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#                 mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#                 mask = cv2.bitwise_or(mask1, mask2)
#             elif target_color == "green":
#                 lower = np.array([47, 67, 66])
#                 upper = np.array([78, 255, 255])
#                 mask = cv2.inRange(hsv, lower, upper)
#             elif target_color == "blue":
#                 lower = np.array([48, 42, 161])
#                 upper = np.array([97, 88, 255])
#                 mask = cv2.inRange(hsv, lower, upper)
#             elif target_color == "yellow":
#                 lower = np.array([16, 51, 157])
#                 upper = np.array([26, 135, 255])
#                 mask = cv2.inRange(hsv, lower, upper)
#             elif target_color == "orange":
#                 lower = np.array([7, 95, 154])
#                 upper = np.array([26, 255, 255])
#                 mask = cv2.inRange(hsv, lower, upper)
#             else:
#                 continue

#             contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             if contours:
#                 largest = max(contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest)
#                 obj_x = x + w // 2
#                 obj_y = y + h // 2
#                 roi = depth_image[y:y+h, x:x+w]
#                 roi = roi[(roi > 100) & (roi < 2000)]
#                 obj_z = int(np.median(roi)) if roi.size > 0 else 0
#                 print(f"[SCAN] Objek ditemukan di arah {direction.upper()} ({obj_x}, {obj_y}, {obj_z})")
#                 return obj_x, obj_y, obj_z

#     speak("Object not found")
#     return None, None, None

def scan_for_object():
    scan_positions = {
        "kanan": 1817,
        "kiri": 2279,
        "atas": 2687,
        "bawah": 2229
    }

    found_object = None

    for direction, pos in scan_positions.items():
        print(f"[SCAN] Menggerakkan kamera ke arah {direction.upper()}")
        if direction in ["kanan", "kiri"]:
            packetHandler.write4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_GOAL_POSITION, pos)
        else:
            packetHandler.write4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_GOAL_POSITION, pos)
        time.sleep(1.5)

        for _ in range(5):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # [Deteksi berdasarkan warna - sama seperti sebelumnya]
            if target_color == "red":
                lower_red1 = np.array([46, 22, 131])
                upper_red1 = np.array([10, 81, 255])
                lower_red2 = np.array([160, 101, 92])
                upper_red2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)
            elif target_color == "green":
                lower = np.array([47, 67, 66])
                upper = np.array([78, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
            elif target_color == "blue":
                lower = np.array([48, 42, 161])
                upper = np.array([97, 88, 255])
                mask = cv2.inRange(hsv, lower, upper)
            elif target_color == "yellow":
                lower = np.array([16, 51, 157])
                upper = np.array([26, 135, 255])
                mask = cv2.inRange(hsv, lower, upper)
            elif target_color == "orange":
                lower = np.array([7, 95, 154])
                upper = np.array([26, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
            else:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours and found_object is None:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                obj_x = x + w // 2
                obj_y = y + h // 2
                roi = depth_image[y:y+h, x:x+w]
                roi = roi[(roi > 100) & (roi < 2000)]
                # obj_z = int(np.median(roi)) if roi.size > 0 else 0
                # found_object = (obj_x, obj_y, obj_z)
                obj_z = int(np.median(roi)) if roi.size > 0 else 0
                if obj_z > 0:
                    found_object = (obj_x, obj_y, obj_z)
                    print(f"[SCAN] Objek ditemukan di arah {direction.upper()} ({obj_x}, {obj_y}, {obj_z})")
                else:
                    print(f"[SCAN] Objek terdeteksi secara visual, tapi depth tidak valid.")


                # print(f"[SCAN] Objek ditemukan di arah {direction.upper()} ({obj_x}, {obj_y}, {obj_z})")
    if found_object:
        return found_object
    else:
        speak("Object not found")
        return None, None, None


# --------------------------- #
# TAHAP 1: TRACK OBJEK       #
# --------------------------- #
print("[INFO] Melakukan scanning awal...")
# obj_x, obj_y, obj_z = scan_for_object()
# if obj_x is None:
#     pipeline.stop()
#     portHandler.closePort()
#     cv2.destroyAllWindows()
#     exit()

obj_x, obj_y, obj_z = scan_for_object()
if None in (obj_x, obj_y, obj_z):
    print("[ERROR] Objek tidak ditemukan. Menghentikan sistem.")
    speak("Object not found. Please try again.")
    pipeline.stop()
    portHandler.closePort()
    cv2.destroyAllWindows()
    exit()

print(f"[INFO] Melanjutkan tracking dari posisi scanning: ({obj_x}, {obj_y}, {obj_z})")

stable_counter = 0
stable_required = 1
start_tracking_time = time.time()
timeout_duration = 2  # dalam detik


while True:
    dxl_pos_x = packetHandler.read4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_PRESENT_POSITION)[0]
    dxl_pos_y = packetHandler.read4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_PRESENT_POSITION)[0]

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    if target_color == "red":
        lower_red1 = np.array([46, 22, 131])
        upper_red1 = np.array([10, 81, 255])
        lower_red2 = np.array([160, 101, 92])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif target_color == "green":
        lower = np.array([47, 67, 66])
        upper = np.array([78, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif target_color == "blue":
        lower = np.array([48, 42, 161])
        upper = np.array([97, 88, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif target_color == "yellow":
        lower = np.array([16, 51, 157])
        upper = np.array([26, 135, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif target_color == "orange":
        lower = np.array([7, 95, 154])
        upper = np.array([26, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        mask = np.zeros_like(hsv[:, :, 0])

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if time.time() - start_tracking_time > timeout_duration:
        print("[TIMEOUT] Objek tidak ditemukan dalam 2 detik.")
        speak("Object not found")
        cv2.putText(color_image, "Object not found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow("Tracking Objek", color_image)
        cv2.waitKey(2000)
        pipeline.stop()
        portHandler.closePort()
        cv2.destroyAllWindows()
        exit()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        object_center_x = x + w // 2
        object_center_y = y + h // 2
        frame_center_x = color_image.shape[1] // 2
        frame_center_y = color_image.shape[0] // 2
        error_x = frame_center_x - object_center_x
        error_y = frame_center_y - object_center_y

        dxl_target_x = dxl_pos_x
        dxl_target_y = dxl_pos_y
        if abs(error_x) > threshold:
            dxl_target_x = apply_limits(int(dxl_pos_x + Kp * error_x), *LIMITS_X)
        if abs(error_y) > threshold:
            dxl_target_y = apply_limits(int(dxl_pos_y + Kp * error_y), *LIMITS_Y)

        if dxl_target_x != dxl_pos_x:
            packetHandler.write4ByteTxRx(portHandler, DXL_ID_HORIZONTAL, ADDR_GOAL_POSITION, dxl_target_x)
        if dxl_target_y != dxl_pos_y:
            packetHandler.write4ByteTxRx(portHandler, DXL_ID_VERTICAL, ADDR_GOAL_POSITION, dxl_target_y)

        roi = depth_image[y:y+h, x:x+w]
        roi = roi[(roi > 100) & (roi < 2000)]
        obj_z = int(np.median(roi)) if roi.size > 0 else 0

        obj_x, obj_y = object_center_x, object_center_y

        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(color_image, f"Cube ({obj_x},{obj_y},{int(obj_z/10)}cm)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print("[INFO] Tracking objek...")

        if abs(error_x) < 10 and abs(error_y) < 10:
            stable_counter += 1
            print(f"[INFO] Objek stabil ({stable_counter}/{stable_required})")
            if stable_counter >= stable_required:
                print("[INFO] Objek dikunci. Siap ke deteksi tangan.")
                # time.sleep(1)
                break
        else:
            stable_counter = 0


    cv2.imshow("Tracking Objek", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
        portHandler.closePort()
        cv2.destroyAllWindows()
        exit()

# multithread
def speak_async(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] Voice output failed: {e}")
    threading.Thread(target=run, daemon=True).start()

speak_async("object is here. grab it!")

# --------------------------- #
# TAHAP 2: DETEKSI TANGAN    #
# --------------------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()
engine.setProperty('rate', 200)
holding_object = False
last_feedback = ""
last_talk_time = 0
last_hand_pos = None

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = spatial.process(temporal.process(frames.get_depth_frame()))

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    feedback = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_z_list = []
            for idx in [0, 5, 9, 13, 17]:
                lm = hand_landmarks.landmark[idx]
                x = int(lm.x * color_image.shape[1])
                y = int(lm.y * color_image.shape[0])
                if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                    z = depth_image[y, x]
                    if 100 < z < 2000:
                        hand_z_list.append(z)

            hand_z = int(np.median(hand_z_list)) if hand_z_list else 0
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_cx = int(middle_mcp.x * color_image.shape[1])
            palm_cy = int(middle_mcp.y * color_image.shape[0])
            last_hand_pos = (palm_cx, palm_cy, hand_z)

            # if palm_cx < obj_x - 30:
            #     feedback += "Move right. "
            # elif palm_cx > obj_x + 30:
            #     feedback += "Move left. "
            # if palm_cy < obj_y - 30:
            #     feedback += "Move down. "
            # elif palm_cy > obj_y + 30:
            #     feedback += "Move up. "

            # X axis (horizontal)
            if obj_x - 30 <= palm_cx < obj_x - 10:
                feedback += "A little bit right. "
            elif palm_cx < obj_x - 30:
                feedback += "Move right. "
            elif obj_x + 10 < palm_cx <= obj_x + 30:
                feedback += "A little bit left. "
            elif palm_cx > obj_x + 30:
                feedback += "Move left. "

            # Y axis (vertical)
            if obj_y - 30 <= palm_cy < obj_y - 10:
                feedback += "A little bit down. "
            elif palm_cy < obj_y - 30:
                feedback += "Move down. "
            elif obj_y + 10 < palm_cy <= obj_y + 30:
                feedback += "A little bit up. "
            elif palm_cy > obj_y + 30:
                feedback += "Move up. "

            selisih_z = hand_z - obj_z
            toleransi_z = max(50, int(obj_z * 0.15))
            # if selisih_z > toleransi_z:
            #     feedback += "Move backward | "
            # elif selisih_z < -toleransi_z:
            #     feedback += "Move forward | "
            # Z axis (depth)
            if toleransi_z < selisih_z <= toleransi_z + 100:
                feedback += "A little bit backward. "
            elif selisih_z > toleransi_z + 100:
                feedback += "Move backward. "
            elif -toleransi_z - 100 <= selisih_z < -toleransi_z:
                feedback += "A little bit forward. "
            elif selisih_z < -toleransi_z - 100:
                feedback += "Move forward. "

            if not feedback and not holding_object:
                feedback = "Grab the object now!"
            elif not feedback and holding_object:
                feedback = "Object successfully grasped."

            if not holding_object and is_hand_closed(hand_landmarks):
                holding_object = True
                print("[INFO] Hand is closed. Object successfully grasped.")
                speak_async("Object successfully grasped") #multithread
                break

            print(f"[DEBUG] Obj Z: {obj_z} | Hand Z: {hand_z} | Selisih: {selisih_z} | Toleransi: {toleransi_z}")
            print("[INFO]", feedback)

            if feedback != last_feedback or (time.time() - last_talk_time) > 2:
                speak_async(feedback) #multithread
                last_feedback = feedback
                last_talk_time = time.time()

            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(color_image, (palm_cx, palm_cy), 5, (255, 255, 0), -1)

    cv2.putText(color_image, f"Target Obj: ({obj_x},{obj_y},{int(obj_z/10)}cm)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if last_hand_pos:
        x, y, z = last_hand_pos
        cv2.putText(color_image, f"Hand Pos: ({x}, {y}, {int(z/10)}cm)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if holding_object:
        cv2.putText(color_image, "Holding object", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        time.sleep(1)
        break

    cv2.imshow("Deteksi Tangan", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
portHandler.closePort()
cv2.destroyAllWindows()