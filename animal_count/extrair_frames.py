import cv2
import os
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================

VIDEOS = [
    {
        "path": "videos/visao_superior.mp4",
        "nome": "superior",
        "quantidade": 200
    },
    {
        "path": "videos/visao_tras.mp4",
        "nome": "tras",
        "quantidade": 200
    }
]


PASTA_SAIDA = "dataset_raw"


def extrair_frames (
    caminho_video,
    nome,
    quantidade
):
    video = cv2.VideoCapture(
        caminho_video
    )

    if not video.isOpened():

        print(f"Erro não foi possível abrir: {caminho_video}")

        return

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    duracao_segundos = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(
        "Total de frames:",
        total_frames
    )

    print(
        "FPS:",
        round(fps, 2)
    )

    print(
        "Duração:",
        round(duracao_segundos, 2),
        "segundos"
    )

    quantidade_real = min(
        quantidade,
        total_frames
    )

    indices = np.linspace(
        0,
        total_frames - 1,
        quantidade_real,
        dtype=int
    )

    indices = np.unique(indices)

    pasta_video = os.path.join(
        PASTA_SAIDA,
        nome
    )


    os.makedirs(
        pasta_video,
        exist_ok=True
    )

    salvas = 0

    for numero, frame_index in enumerate(
        indices
    ):
        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        sucesso, frame = video.read()

        if not sucesso:
            print(f"Não foi possível ler frame {frame_index}")
            continue

        nome_arquivo = (
            f"{nome}_"
            f"{numero:04d}_"
            f"frame_{frame_index:06d}.jpg"
        )


        caminho_saida = os.path.join(
            pasta_video,
            nome_arquivo
        )

        # ----------------------------------------------------
        # Salva imagem
        # ----------------------------------------------------

        cv2.imwrite(
            caminho_saida,
            frame
        )


        salvas += 1


    video.release()


    print(
        f"Finalizado: {salvas} imagens salvas."
    )

    print(
        f"Pasta: {pasta_video}"
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

os.makedirs(
    PASTA_SAIDA,
    exist_ok=True
)


for config in VIDEOS:
    extrair_frames(
        caminho_video=config["path"],
        nome=config["nome"],
        quantidade=config["quantidade"]
    )

print("\n========================================")
print("Extração concluída!")
print("========================================")

