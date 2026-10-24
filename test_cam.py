import cv2

for device in ["/dev/video0", "/dev/video1"]:
    print(f"\n--- Test {device} avec V4L2 ---")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    print("Ouvert :", cap.isOpened())
    if cap.isOpened():
        ret, frame = cap.read()
        print("Frame lue :", ret)
        if ret:
            print("Dimensions :", frame.shape)
        cap.release()
