from ultralytics import YOLO


model = YOLO(
    "../models/counter/yolo26s.pt"
)

model.train(
    data="../dataset_raw/My First Project.yolo26/data.yaml",
    epochs=1,
    imgsz=960,
    patience=15
)