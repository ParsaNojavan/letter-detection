import cv2
import mediapipe as mp
import numpy as np
import math
import os

def find_degree(base_point, second_point):
    dx = base_point[0] - second_point[0]
    dy = base_point[1] - second_point[1]

    theta = np.degrees(np.arctan2(dy, dx))
    theta = theta if theta > 0 else theta + 360
    return theta

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# مسیر مدل (پشتیبانی از هر دو مسیر جاری و نسبی)
MODEL_PATH = "./hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# رسم دست
# ============================================================
def draw_hand_landmarks(image, landmarks):
    h, w, _ = image.shape

    for i, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(
            image,
            str(i),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for start, end in connections:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)

        cv2.line(image, (x1, y1), (x2, y2), (255, 120, 0), 2)

# ============================================================
# تشخیص وضعیت و تعداد انگشت‌ها (مقاوم در برابر چرخش)
# ============================================================
def count_fingers(hand_landmarks):
    points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
    wrist = points[0]

    # بررسی باز بودن انگشتان با مقایسه فاصله نوک نسبت به بند مفصل
    index_open  = dist(points[8], wrist)  > dist(points[6], wrist) * 1.12
    middle_open = dist(points[12], wrist) > dist(points[10], wrist) * 1.12
    ring_open   = dist(points[16], wrist) > dist(points[14], wrist) * 1.12
    pinky_open  = dist(points[20], wrist) > dist(points[18], wrist) * 1.12
    thumb_open  = dist(points[4], points[17]) > dist(points[2], points[17]) * 1.25

    finger_count = sum([index_open, middle_open, ring_open, pinky_open, thumb_open])
    return finger_count, points

# ============================================================
# موتور تشخیص اختصاصی ۱۰ حرف
# ============================================================
def recognize_character(points):
    wrist = points[0]
    scale = dist(points[0], points[9])
    if scale == 0:
        scale = 1.0

    # وضعیت عمودی/باز بودن هر انگشت
    index_up  = dist(points[8], wrist)  > dist(points[6], wrist) * 1.12
    middle_up = dist(points[12], wrist) > dist(points[10], wrist) * 1.12
    ring_up   = dist(points[16], wrist) > dist(points[14], wrist) * 1.12
    pinky_up  = dist(points[20], wrist) > dist(points[18], wrist) * 1.12
    thumb_out = dist(points[4], points[17]) > dist(points[2], points[17]) * 1.25

    # فواصل کلیدی نرمال‌شده
    d_thumb_index  = dist(points[4], points[8]) / scale
    d_index_middle = dist(points[8], points[12]) / scale
    d_middle_ring  = dist(points[12], points[16]) / scale

    horiz_index  = abs(points[8][0] - points[5][0]) > abs(points[8][1] - points[5][1]) * 1.3
    horiz_middle = abs(points[12][0] - points[9][0]) > abs(points[12][1] - points[9][1]) * 1.3
    pointing_side = abs(points[8][0] - points[0][0]) > 0.15

    is_back_of_hand = points[9][2] < points[0][2]


    # ----------------------------------------------------
    # 1. م (M) - اشاره رو به پایین و بقیه مشت
    # ----------------------------------------------------
    if points[8][1] > points[5][1] and not middle_up and not ring_up and not pinky_up:
        if (points[8][1] - points[0][1]) > 0.05:  # نوک اشاره پایین‌تر از مچ
            return "M"

    # ----------------------------------------------------
    # 2. ای (I / Ye) - فقط شست و انگشت کوچک باز (Hang-loose)
    # ----------------------------------------------------
    if pinky_up and thumb_out and not index_up and not middle_up and not ring_up:
        return "I"

    # ----------------------------------------------------
    # 3. ا O (Alef ba Seda-ye O) - حلقه شست و اشاره، ولی ۳ انگشت دیگر باز به بالا
    # ----------------------------------------------------
    if index_up and middle_up and thumb_out and not ring_up and not pinky_up:
        return "O"
    # ----------------------------------------------------
    # 4. آ (A ba Kolah) - حالت L شکل (اشاره بالا + شست کاملاً باز به بغل)
    # ----------------------------------------------------
    if index_up and thumb_out and d_thumb_index > 0.42 and not middle_up and not ring_up and not pinky_up:
        return "A"

    # ----------------------------------------------------
    # 5,6. ع (Ayn) - اشاره و میانی به صورت افقی رو به جلو
    # ----------------------------------------------------
    # اختلاف ارتفاع اندک اما فاصله x زیاد، و هر دو موازی
    if horiz_index and horiz_middle and pointing_side and not ring_up and not pinky_up:
        # فاصله عمودی بین دو انگشت (مهم‌ترین فاکتور در حالت افقی)
        vertical_gap = abs(points[8][1] - points[12][1]) / scale

        # اگر دو انگشت به هم چسبیده باشند:
        # فاصله اقلیدسی نرمال کمتر از 0.38 و اختلاف ارتفاع ناچیز است
        if d_index_middle < 0.38 and vertical_gap < 0.28:
            return "Ayn"
        else:
            return "Ghayn"

    # ----------------------------------------------------
    # 7. ن (Noon) - انگشتان اشاره و میانی چسبیده به هم کاملاً عمودی بالا
    # ----------------------------------------------------
    if is_back_of_hand and index_up and middle_up and ring_up and pinky_up and not thumb_out:
        return "Noon"

    # ----------------------------------------------------
    # 8. ا A (Alef sade) - ۴ انگشت چسبیده و کاملاً کشیده رو به بالا، شست بسته
    # ----------------------------------------------------
    if index_up and middle_up and ring_up and pinky_up and not thumb_out:
        if d_index_middle < 0.22 and d_middle_ring < 0.22:
            return "Alef"

    # ----------------------------------------------------
    # 9. گ G - دو انگشت اشاره و وسط بالا و پشت دست به دوربین
    # ----------------------------------------------------
    if is_back_of_hand and index_up and middle_up and not ring_up and not pinky_up and not thumb_out:
        # فاصله‌ی بین نوک انگشت اشاره (8) و وسط (12)
        d_index_middle = dist(points[8], points[12]) / scale

        # در حرف «گ» دو انگشت اشاره و وسط معمولاً جدا از هم (یا با فاصله مشخص) هستند
        # اگر می‌خواهی در هر دو حالت (چسبیده یا باز) بگیرد، شرط d_index_middle را بردار، 
        # اما برای تشخیص دقیق‌تر، محدوده فاصله زیر بسیار مناسب است:
        if d_index_middle > 0.15:
            return "Gaf"

    # ----------------------------------------------------
    # 10. ه H - تمامی انگشت ها بسته
    # ----------------------------------------------------
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_out:
        return "H"

    # ----------------------------------------------------
    # 11. ط Ta - تمامی انگشت ها بسته اشاره باز
    # ----------------------------------------------------
    if index_up and not middle_up and not ring_up and not pinky_up:
        # ۱. بررسی اینکه اشاره واقعاً صاف و کشیده رو به بالا است (تفاوت y نوک با بند پایین)
        index_straight = points[8][1] < points[6][1] < points[5][1]
        
        # ۲. مطمئن شویم نوک شست به انگشت وسط یا حلقه چسبیده/نزدیک است (شست بسته)
        d_thumb_middle = dist(points[4], points[10]) / scale

        if index_straight and d_thumb_middle < 0.35:
            return "Ta"

    # ----------------------------------------------------
    # 12. ه H - تمامی انگشت ها بسته
    # ----------------------------------------------------
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_out:
        return "H"
    

    return "?"

# ============================================================
# شروع دوربین
# ============================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("خطا: دوربین باز نشد.")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("خطا در دریافت تصویر از دوربین")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = landmarker.detect(mp_image)

        finger_count = 0
        degree = 90
        char1 = "?"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            finger_count, points = count_fingers(hand_landmarks)
            degree = int(find_degree(points[0], points[12]))
            char1 = recognize_character(points)
            draw_hand_landmarks(frame, hand_landmarks)

        # نمایش خروجی
        text = f"{char1}"
        cv2.putText(
            frame,
            text,
            (35, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )

        cv2.imshow("MediaPipe Hand Tracking", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
