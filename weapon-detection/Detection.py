from ultralytics import YOLO
import cv2
import os

# ✅ Aapka trained model load karo
model = YOLO(r'runs/detect/weapon_detector/weights/best.pt')

# ✅ Test images list
test_images = [
    "test/army.jpg",
    "test/terrorists.jpg",
    "test/terrorists2.jpg",
    "test/weapons.jpg",
    "test/weapons2.jpg",
]

# ✅ Ek ek image test karo
for image_path in test_images:
    if not os.path.exists(image_path):
        print(f"❌ File nahi mili: {image_path}")
        continue

    print(f"\n🔍 Testing: {image_path}")

    results = model(
        image_path,
        conf=0.60,      # 60% confidence threshold
        iou=0.45,
        imgsz=640,
    )

    for result in results:
        boxes = result.boxes
        print(f"   ✅ {len(boxes)} object(s) detected:")

        for box in boxes:
            cls_id = int(box.cls[0])
            conf  = float(box.conf[0])
            label = model.names[cls_id]
            print(f"      → {label}: {conf:.2%} confidence")

        # Result save karo
        save_name = "output_" + os.path.basename(image_path)
        result.save(filename=save_name)
        print(f"   📸 Saved: {save_name}")

        # Display karo
        img = cv2.imread(save_name)
        cv2.imshow(f"Detection - {os.path.basename(image_path)}", img)
        print("   ⏎  Koi bhi key dabao agli image ke liye...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("\n✅ Saari images test ho gayi!")