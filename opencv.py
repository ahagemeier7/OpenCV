import cv2

camera = cv2.VideoCapture(0)

while True:
  sucesso, frame = camera.read()

  cv2.line(
    frame,
    (0, 300),
    (1280, 300),
    (0, 255, 0),
    3
  )
  
  if not sucesso:
    break
  
  cv2.imshow("Camera",frame)
  
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
  
  altura, largura, canais = frame.shape

  print("Largura:", largura)
  print("Altura:", altura)
  

  
camera.release()
cv2.destroyAllWindows()

