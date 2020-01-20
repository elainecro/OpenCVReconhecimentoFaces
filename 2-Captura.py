import cv2
import numpy as np

classificador = cv2.CascadeClassifier("cascades\\haarcascade-frontalface-default.xml")
classificadorOlhos = cv2.CascadeClassifier("cascades\\haarcascade-eye.xml")

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
largura, altura = 220, 220

id = input('Digite seu identificador: ')
print('Capturando as faces...')

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    print(np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiaoFace = imagem[y:y + a, x:x + l]
        regiaoFaceCinza = cv2.cvtColor(regiaoFace, cv2.COLOR_BGR2GRAY)

        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoFaceCinza)

        for (xO, yO, lO, aO) in olhosDetectados:
            cv2.rectangle(regiaoFace, (xO, yO), (xO + lO, yO + aO), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 110:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x+l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if amostra >= numeroAmostras + 1:
        break

print("Faces capturadas com sucesso")
camera.release()
cv2.destroyAllWindows()