import cv2

dectector = cv2.CascadeClassifier("cascades\\haarcascades-frontalface-default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read("classificadorEigen.yml")

camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
