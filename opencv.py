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

# Texto que já foi confirmado
letras_confirmadas = []

# Letra que já ficou estável por alguns frames
letra_estavel = None

# Letra que estamos verificando neste momento
letra_candidata = None

# Quantos frames seguidos estamos vendo a mesma letra
frames_candidata = 0

# Quantos frames estamos sem detectar uma mão
frames_sem_mao = 0


# Quantos frames iguais são necessários para considerar
# a previsão estável
FRAMES_PARA_CONFIRMAR = 10

# Quantos frames sem mão são necessários
# para confirmar/adicionar a letra ao texto
FRAMES_SEM_MAO_PARA_SALVAR = 10


CAMINHO_MODELO = "models/modelo_libras.pkl"
ARQUIVO_DATASET = "dataset.csv"


# ============================================================
# TECLAS PARA COLETA DE DATASET
# ============================================================

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
    ord("l"): "L",
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

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


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


# ============================================================
# EXTRAIR FEATURES
# ============================================================

def extrair_features(hand):

    # Landmark 0 = pulso
    wrist = hand[0]

    # Base do indicador
    index_base = hand[5]

    # Base do mindinho
    little_base = hand[17]


    # ========================================================
    # TAMANHO DE REFERÊNCIA DA MÃO
    # ========================================================

    escala = distancia3d(
        index_base,
        little_base
    )


    # Evita divisão por zero
    if escala < 0.000001:
        return None


    features = []


    # ========================================================
    # TRANSFORMANDO 21 LANDMARKS EM 63 FEATURES
    # ========================================================

    for point in hand:

        # Colocamos o pulso como origem
        # e dividimos pelo tamanho da mão

        x = (
            point.x - wrist.x
        ) / escala

        y = (
            point.y - wrist.y
        ) / escala

        z = (
            point.z - wrist.z
        ) / escala


        features.extend([
            x,
            y,
            z
        ])


    return features


# ============================================================
# CRIAR DATASET
# ============================================================

def criar_dataset():

    with open(
        ARQUIVO_DATASET,
        "w",
        newline=""
    ) as arquivo:

        escritor = csv.writer(arquivo)

        cabecalho = []


        # 21 pontos
        for i in range(21):

            cabecalho.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])


        cabecalho.append("label")

        escritor.writerow(
            cabecalho
        )


    print("Novo dataset criado!")


# ============================================================
# SALVAR AMOSTRA
# ============================================================

def salvar_amostra(features, label):

    with open(
        ARQUIVO_DATASET,
        "a",
        newline=""
    ) as arquivo:

        escritor = csv.writer(
            arquivo
        )

        linha = (
            features + [label]
        )

        escritor.writerow(
            linha
        )


    print(
        f"Amostra salva: {label}"
    )


# ============================================================
# CRIAR DATASET CASO NÃO EXISTA
# ============================================================

if not os.path.exists(
    ARQUIVO_DATASET
):

    criar_dataset()


# ============================================================
# ABRINDO CÂMERA
# ============================================================

camera = cv2.VideoCapture(0)


# ============================================================
# CRIANDO HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    while True:


        # ====================================================
        # CAPTURANDO FRAME
        # ====================================================

        sucesso, frame = camera.read()


        if not sucesso:
            break


        # A cada frame começamos sem features
        features_atuais = None


        # ====================================================
        # PREPARANDO IMAGEM PARA MEDIAPIPE
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
        # DETECÇÃO DAS MÃOS
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


            # Se entrou aqui, existe uma mão.
            #
            # Então zeramos o contador de
            # frames sem mão.
            frames_sem_mao = 0


            # =================================================
            # EXTRAINDO FEATURES
            # =================================================

            features = extrair_features(
                hand
            )


            if features is not None:


                # Guardamos para poder continuar
                # coletando dataset manualmente
                features_atuais = features


                # =============================================
                # FAZENDO PREVISÃO
                # =============================================

                if modelo is not None:


                    previsao = modelo.predict(
                        [features]
                    )[0]


                    # =========================================
                    # ESTABILIZAÇÃO DA PREVISÃO
                    # =========================================

                    # Continua vendo a mesma letra?
                    if previsao == letra_candidata:

                        frames_candidata += 1


                    # Mudou a previsão?
                    else:

                        letra_candidata = previsao

                        frames_candidata = 1


                    # =========================================
                    # LETRA FICOU ESTÁVEL
                    # =========================================

                    if (
                        frames_candidata
                        >= FRAMES_PARA_CONFIRMAR
                    ):

                        letra_estavel = (
                            letra_candidata
                        )


                    # =========================================
                    # MOSTRANDO PREVISÃO BRUTA
                    # =========================================

                    cv2.putText(

                        frame,

                        f"Sinal: {previsao}",

                        (30, 50),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        (255, 0, 0),

                        2
                    )


            # =================================================
            # DESENHANDO OS LANDMARKS
            # =================================================

            for index, point in enumerate(
                hand
            ):


                x = int(
                    point.x * w
                )

                y = int(
                    point.y * h
                )


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
        # VERIFICANDO SE A MÃO FOI RETIRADA
        # ====================================================

        if len(
            result.hand_landmarks
        ) == 0:


            frames_sem_mao += 1


            # Ficou sem mão durante
            # frames suficientes?
            if (
                frames_sem_mao
                >= FRAMES_SEM_MAO_PARA_SALVAR
            ):


                # Existe uma letra estável esperando
                # confirmação?
                if letra_estavel is not None:


                    # =========================================
                    # ADICIONANDO LETRA AO TEXTO
                    # =========================================

                    letras_confirmadas.append(
                        letra_estavel
                    )


                    print(
                        "Letra confirmada:",
                        letra_estavel
                    )


                    print(
                        "Texto:",
                        "".join(
                            letras_confirmadas
                        )
                    )


                    # =========================================
                    # RESET PARA PRÓXIMA LETRA
                    # =========================================

                    letra_estavel = None

                    letra_candidata = None

                    frames_candidata = 0


                # Reinicia contagem sem mão
                frames_sem_mao = 0


        # ====================================================
        # TEXTO FORMADO
        # ====================================================

        texto_formado = "".join(
            letras_confirmadas
        )


        cv2.putText(

            frame,

            f"Texto: {texto_formado}",

            (30, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            2
        )


        # ====================================================
        # LETRA QUE JÁ ESTÁ ESTÁVEL
        # ====================================================

        if letra_estavel is not None:


            cv2.putText(

                frame,

                f"Confirmar: {letra_estavel}",

                (30, 150),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2
            )


        # ====================================================
        # INSTRUÇÕES
        # ====================================================

        cv2.putText(

            frame,

            "Tire a mao para confirmar | Q=Sair",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1
        )


        # ====================================================
        # MOSTRANDO CÂMERA
        # ====================================================

        cv2.imshow(
            "Camera",
            frame
        )


        # ====================================================
        # TECLADO
        # ====================================================

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ====================================================
        # SAIR
        # ====================================================

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
        # SALVAR NOVAS AMOSTRAS NO DATASET
        # ====================================================

        if tecla in LABELS:


            if features_atuais is not None:


                label = LABELS[
                    tecla
                ]


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