from ultralytics import YOLO
import cv2

model = YOLO(r'runs/detect/weapon_detector/weights/best.pt')

print("🎥 Webcam shuru ho raha hai...")
print("❌ Band karne ke liye 'Q' dabao")

# Webcam kholo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam nahi khuli! source=1 try karo")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame nahi mila!")
        break

    # Detection karo
    results = model(frame, conf=0.40, iou=0.45, imgsz=640, verbose=False)

    # Frame pe boxes draw karo
    annotated_frame = results[0].plot()

    # Detection print karo
    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  🔫 {label}: {conf:.2%}")

    # Window mein dikhao
    cv2.imshow("🔫 Weapon Detection - Q dabao band karne ke liye", annotated_frame)

    # Q dabane pe band karo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ Band ho raha hai...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam band ho gayi!")