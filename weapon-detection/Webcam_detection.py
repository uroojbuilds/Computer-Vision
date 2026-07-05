from ultralytics import YOLO
import cv2
import winsound
import threading

weapon_model = YOLO(r'runs/detect/weapon_detector/weights/best.pt')
coco_model = YOLO('yolov8n.pt')

def play_alarm():
    for _ in range(3):
        winsound.Beep(1000, 500)

print("🎥 Webcam shuru ho raha hai...")
print("❌ Band karne ke liye 'Q' dabao")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Webcam nahi khuli!")
    exit()

print("✅ Webcam khul gayi! Detection shuru...")

prev_labels = set()
alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    weapon_results = weapon_model(
        frame, conf=0.50, iou=0.45, imgsz=640,
        verbose=False, stream=False
    )

    coco_results = coco_model(
        frame, conf=0.30, iou=0.45, imgsz=640,
        verbose=False, stream=False, classes=[43]
    )

    annotated_frame = weapon_results[0].plot()
    current_labels = set()
    knife_detected = False

    for box in coco_results[0].boxes:
        if int(box.cls[0]) == 43:
            knife_detected = True
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Knife {conf:.0%}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            current_labels.add(f"Knife: {conf:.0%}")

    for box in weapon_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = weapon_model.names[cls_id]
        if label == "Automatic Rifle" and knife_detected:
            continue
        current_labels.add(f"{label}: {conf:.0%}")

    if current_labels != prev_labels:
        if current_labels:
            print("\n🚨 ALARM — WEAPON DETECTED! 🚨")
            for l in current_labels:
                print(f"  🔫 {l}")
            # Alarm sirf ek baar bajao
            if not alarm_playing:
                alarm_playing = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            print("  ✅ Weapon hataya gaya — Safe")
            alarm_playing = False
        prev_labels = current_labels

    # Screen pe status
    if current_labels:
        status = "🚨 WEAPON DETECTED!"
        color = (0, 0, 255)
    else:
        status = "✅ Safe"
        color = (0, 255, 0)

    cv2.putText(annotated_frame, status, (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Weapon Detection - Q dabao", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ Band ho raha hai...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done!")