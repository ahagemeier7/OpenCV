import cv2
from ultralytics import YOLO

model = YOLO("../runs/detect/train-2/weights/best.pt") #Modelo treinado com base nas imagens 

video = cv2.VideoCapture("videos/visao_superior.mp4")



while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(
        frame,
        classes=[0],#Classe 0 - sheep definida no roboflow
        conf=0.50,
        imgsz=960,
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