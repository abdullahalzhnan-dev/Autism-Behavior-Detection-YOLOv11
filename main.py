from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

# تحميل الموديل
model = YOLO("best1.pt")

# تشغيل الكاميرا
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

alarm_playing = False

def play_alarm():
    global alarm_playing
    while alarm_playing:
        playsound("warning 1000283055.mp3")

frame_count = 0
detect_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 تحليل كل 3 فريمات (temporal)
    if frame_count % 3 != 0:
        continue

    results = model.predict(
        frame,
        conf=0.5,
        imgsz=480,
        verbose=False
    )

    boxes = results[0].boxes

    # 🔥 إذا في اكتشاف
    if len(boxes) > 0:
        detect_counter += 1
    else:
        detect_counter = 0

    # 🔥 شرط الإنذار (تجنب false positives)
    if detect_counter >= 2:
        if not alarm_playing:
            alarm_playing = True
            threading.Thread(target=play_alarm, daemon=True).start()
    else:
        alarm_playing = False

    # رسم النتائج
    frame = results[0].plot()

    cv2.imshow("Weapon Detection System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()