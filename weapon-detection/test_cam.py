import cv2

print("Webcam test kar raha hai...")

# 0, 1, 2 try karo
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera {i} mila!")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)  # 3 second dikhao
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"❌ Camera {i} nahi mila")