import cv2
from ultralytics import YOLO


CAMINHO_MODELO = "models/counter/yolo26s.pt"
CAMINHO_VIDEO = "videos/lvl_1.mp4"

# Escolha um frame onde ele esteja errando bastante
FRAME_ALVO = 100


model = YOLO(CAMINHO_MODELO)

print("Classes do modelo:")
print(model.names)


# Descobre automaticamente o ID de sheep
sheep_id = None

for id_classe, nome in model.names.items():
    if nome == "sheep":
        sheep_id = id_classe
        break

print("ID de sheep:", sheep_id)


video = cv2.VideoCapture(CAMINHO_VIDEO)

video.set(
    cv2.CAP_PROP_POS_FRAMES,
    FRAME_ALVO
)

success, frame = video.read()

if not success:
    print("Não consegui carregar o frame.")
    exit()


def testar(
    nome,
    classes=None,
    conf=0.05,
    imgsz=640
):

    print("\n==============================")
    print(nome)
    print("==============================")

    results = model.predict(
        frame,
        classes=classes,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )

    result = results[0]

    print(
        "Total de caixas:",
        len(result.boxes)
    )


    for box in result.boxes:

        classe_id = int(
            box.cls[0]
        )

        confianca = float(
            box.conf[0]
        )

        nome_classe = model.names[
            classe_id
        ]

        print(
            f"{nome_classe}: {confianca:.3f}"
        )


    frame_drawn = result.plot()

    cv2.imwrite(
        f"{nome}.jpg",
        frame_drawn
    )


# TESTE 1
# O que o YOLO realmente está enxergando?
testar(
    nome="debug_1_todas_classes",
    classes=None,
    conf=0.05,
    imgsz=640
)


# TESTE 2
# Ele enxerga as ovelhas mas com confiança baixa?
testar(
    nome="debug_2_sheep_640",
    classes=[sheep_id],
    conf=0.05,
    imgsz=640
)


# TESTE 3
# Aumentar resolução de inferência ajuda?
testar(
    nome="debug_3_sheep_1280",
    classes=[sheep_id],
    conf=0.05,
    imgsz=1280
)


video.release()

cv2.destroyAllWindows()