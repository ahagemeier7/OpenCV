from ultralytics import YOLO


model = YOLO(
    "../models/counter/yolo26s.pt"
)

model.train(
    data="../dataset_raw/My First Project.yolo26/data.yaml",
    epochs=20,#Quantidade de vezes que ele vai olhar o dataset
    imgsz=640,
    patience=8#Caso o modelo não melhore em x vezes ele para de treinar para não dar overfitting
)