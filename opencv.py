import cv2
import time
import mediapipe as mp
import math

# Encurtando nomes das configs
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

#Abrindo a primeira câmera disponivel no pc
camera = cv2.VideoCapture(0)

#Criando o landmarker
with HandLandmarker.create_from_options(options) as landmarker:

  while True:
    sucesso, frame = camera.read()
    
    if not sucesso:
      break
    
    #Convertendo o frame para cores para o media pipe entender
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    #Convertendo o frame em um objeto do media pipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    
    #Declarando o tempo para o media pipe entender a ordem dos frames
    timestamp_ms = int(time.monotonic() * 1000)
    
    #Encontra a região das mãoes no vídeo
    result = landmarker.detect_for_video(
      mp_image,
      timestamp_ms
    )
    
    #Pegando a altura e largura do frame
    h, w, _ = frame.shape
    
    #PErcorrendo os pontos de cada uma das mãos
    for hand in result.hand_landmarks:
      
      fist = hand[0]
      thumb = hand[4]
      index_f = hand[8]
      middle = hand[12]
      ring = hand[16]
      little = hand[20]
      
      distancia = math.sqrt(
        (index_f.x - fist.x) ** 2 + (index_f.y - fist.y) ** 2
      )
      
      print(distancia)
      
      for index,point in enumerate(hand):
        #Convertendo as coordenadas devolvidas pelo mediapipe em pixels
        # Mediapipe retorna cordenadas normalizadas ou seja vai de 0 a 1 (Como 0 a 100% da tela, ai se multiplica pelos 
        # pixels, pega a posição certa)
        x = int(point.x * w)
        y = int(point.y * h)

                
        #Desenhando o circulo na mão
        cv2.circle(
          frame,
          (x,y),
          5, # Raio do circulo em pixels
          (0,255,0), #Cor
          -1 #Circulo deve ser preenchido
        )
        
        cv2.putText(
          frame,
          str(index),
          (x + 5,y - 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0,0,255),
          1
        )
    
    cv2.imshow("Camera",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  
camera.release()
cv2.destroyAllWindows()

