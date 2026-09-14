import cv2
from ultralytics import YOLO

model = YOLO("../models/counter/yolo26n.pt")

video = cv2.VideoCapture("videos/lvl_1.mp4")



while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(
        frame,
        classes=[18],
        conf=0.40,
        verbose=False
    )

    result = results[0]

    frame_drawn = result.plot()

    cv2.imshow("Frame", frame_drawn)
    key = cv2.waitKey(1)
    if key == 27:
        break

#camera.release()
cv2.destroyAllWindows()