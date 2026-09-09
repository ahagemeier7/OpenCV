import cv2
import time
import mediapipe as mp
import math
import csv
import os
import joblib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

letters_predicted = []
CAMINHO_MODELO = "models/modelo_libras.pkl"
ARQUIVO_DATASET = "dataset.csv"

LABELS = {
    ord("1"): "A",
    ord("2"): "B",
    ord("3"): "C",
    ord("4"): "D",
    ord("5"): "E",
    ord("6"): "F",
    ord("7"): "G",
    ord("8"): "H",
    ord("9"): "I",
    ord("w"): "J",
    ord("e"): "K",
    ord("r"): "L",
    ord("t"): "M",
    ord("y"): "N",
    ord("u"): "O",
    ord("i"): "P",
    ord("o"): "Q",
    ord("p"): "R",
    ord("a"): "S",
    ord("s"): "T",
    ord("d"): "U",
    ord("f"): "V",
    ord("g"): "W",
    ord("h"): "X",
    ord("j"): "Y",
    ord("k"): "Z",
    
}


# ============================================================
# CARREGANDO MODELO DE CLASSIFICAÇÃO
# ============================================================

if os.path.exists(CAMINHO_MODELO):
    modelo = joblib.load(CAMINHO_MODELO)
    print("Modelo carregado!")
else:
    modelo = None
    print("Modelo ainda não existe.")
    print("Você ainda pode coletar dados normalmente.")


# ============================================================
# CONFIGURAÇÃO DO MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


# ============================================================
# FUNÇÕES
# ============================================================

def distancia3d(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def extrair_features(hand):

    wrist = hand[0]

    index_base = hand[5]
    little_base = hand[17]

    # Referência do tamanho da mão
    escala = distancia3d(
        index_base,
        little_base
    )

    if escala < 0.000001:
        return None

    features = []

    for point in hand:

        # Coordenadas relativas ao pulso
        # e normalizadas pelo tamanho da mão

        x = (point.x - wrist.x) / escala
        y = (point.y - wrist.y) / escala
        z = (point.z - wrist.z) / escala

        features.extend([
            x,
            y,
            z
        ])

    return features


def criar_dataset():

    with open(
        ARQUIVO_DATASET,
        "w",
        newline=""
    ) as arquivo:

        escritor = csv.writer(arquivo)

        cabecalho = []

        for i in range(21):

            cabecalho.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        cabecalho.append("label")

        escritor.writerow(cabecalho)

    print("Novo dataset criado!")


def salvar_amostra(features, label):

    with open(
        ARQUIVO_DATASET,
        "a",
        newline=""
    ) as arquivo:

        escritor = csv.writer(arquivo)

        linha = features + [label]

        escritor.writerow(linha)

    print(f"Amostra salva: {label}")


# ============================================================
# CRIAR DATASET CASO NÃO EXISTA
# ============================================================

if not os.path.exists(ARQUIVO_DATASET):
    criar_dataset()


# ============================================================
# ABRINDO CÂMERA
# ============================================================

camera = cv2.VideoCapture(0)


# ============================================================
# CRIANDO HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            break


        # IMPORTANTE:
        # A cada frame começamos sem nenhuma feature.
        # Assim não salvamos acidentalmente a mão de um frame antigo.
        features_atuais = None


        # ====================================================
        # PREPARANDO IMAGEM PARA O MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            time.monotonic() * 1000
        )


        # ====================================================
        # DETECÇÃO DA MÃO
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        h, w, _ = frame.shape


        # ====================================================
        # PROCESSANDO MÃO DETECTADA
        # ====================================================

        for hand in result.hand_landmarks:

            features = extrair_features(hand)

            if features is not None:

                # Guardamos as features atuais
                # para poder salvar pressionando 1, 2 ou 3
                features_atuais = features


                # ============================================
                # PREVISÃO DO MODELO
                # ============================================

                if modelo is not None:

                    previsao = modelo.predict(
                        [features]
                    )[0]

                    cv2.putText(
                        frame,
                        f"Sinal: {previsao}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )
                    
                    
                    
                    letters_predicted.append(previsao)


            # =================================================
            # DESENHANDO LANDMARKS
            # =================================================

            for index, point in enumerate(hand):

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(index),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )


        # ====================================================
        # INSTRUÇÕES NA TELA
        # ====================================================

        cv2.putText(
            frame,
            "1=A  2=B  3=C  R=Reset dataset  Q=Sair",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )


        cv2.imshow(
            "Camera",
            frame
        )


        # ====================================================
        # TECLADO
        # ====================================================

        tecla = cv2.waitKey(1) & 0xFF


        # Sair
        if tecla == ord("q"):
            break


        # ====================================================
        # RECRIAR DATASET
        # ====================================================

        if tecla == ord("r"):

            criar_dataset()

            print(
                "Dataset recriado."
            )

            continue


        # ====================================================
        # SALVAR NOVA AMOSTRA
        # ====================================================

        if tecla in LABELS:

            if features_atuais is not None:

                label = LABELS[tecla]

                salvar_amostra(
                    features_atuais,
                    label
                )

            else:

                print(
                    "Nenhuma mão detectada!"
                )


# ============================================================
# FINALIZAÇÃO
# ============================================================

camera.release()
cv2.destroyAllWindows()