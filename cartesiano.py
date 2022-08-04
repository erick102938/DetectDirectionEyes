import time
from time import sleep

import numpy as np
import tensorflow as tf
import keras as kr
import cv2
def isOpen(number):
    if number > 0.55:
        return 'abierto'
    else:
        return  'cerrado'

def getleftmosteye(eyes):
    leftmost=9999999
    leftmostindex=-1
    for i in range(0,2):
        if eyes[i][0]<leftmost:
            leftmost=eyes[i][0]
            leftmostindex=i
    return eyes[leftmostindex]
def getrightmosteye(eyes):
    rightmost=0
    rightmostindex=-1
    for i in range(0,2):
        if eyes[i][0]>rightmost:
            rightmost=eyes[i][0]
            rightmostindex=i
    return eyes[rightmostindex]


#resolutio 536.7 x 416.33
w = 1920
h = 1080
eyeGaze = kr.models.load_model('modelOnlyh4.h5')
blink = kr.models.load_model('modelBlink.h5')
vid = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
t0 = time.time()
toeyes = time.time()
cara= []
promedioX = []
promedioY = []
contador = 0
direccion = ""
auxDireccion = ""
tCerradoInicio = time.time()
while (True):
    screen = np.zeros((h, w))
    ret, frame = vid.read()
    img0 = []
    img1 = []
    img2 = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if (len(faces) == 1):
        roi_gray = gray[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        roi_color = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        img0 = roi_color
        #cv2.rectangle(frame, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]),(255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if (len(eyes) == 2):

            eyeleft = getleftmosteye(eyes)
            eyeright = getrightmosteye(eyes)

            eyeleft = roi_color[eyeleft[1]:eyeleft[1] + eyeleft[3], eyeleft[0]:eyeleft[0] + eyeleft[2]]
            eyeright = roi_color[eyeright[1]:eyeright[1] + eyeright[3], eyeright[0]:eyeright[0] + eyeright[2]]
            img1 = eyeleft
            img2 = eyeright


            img0 = np.array(img0)
            img1 = np.array(img1)
            img2 = np.array(img2)

            eyeLeft = cv2.resize(img1, (224, 224),interpolation = cv2.INTER_AREA)
            eyeLeft = np.reshape(eyeLeft, [1, 224, 224, 3])
            eyeRight = cv2.resize(img2, (224, 224),interpolation = cv2.INTER_AREA)
            eyeRight = np.reshape(eyeRight, [1, 224, 224, 3])


            img0 = cv2.resize(img0, (224, 224),interpolation = cv2.INTER_AREA)
            img0 = np.reshape(img0, [1, 224, 224, 3])
            img1 = cv2.resize(img1, (180, 180),interpolation = cv2.INTER_AREA)
            img1 = np.reshape(img1, [1, 180, 180, 3])
            img2 = cv2.resize(img2, (180, 180),interpolation = cv2.INTER_AREA)
            img2 = np.reshape(img2, [1, 180, 180, 3])
            predicted = eyeGaze.predict([img0, img1, img2])
            predicted = predicted


            t1 = time.time()
            if (t1 - t0 > 3 and t1 - t0 < 10):
                xPredict = predicted[0][0]
                yPredict = predicted[0][1]
                image = cv2.putText(screen, 'Seteando vista', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,  cv2.LINE_AA)
                promedioX.append(predicted[0][0])
                promedioY.append(predicted[0][1])

                print('dato para promediar: ', np.mean(promedioX), " : ", np.mean(promedioY), " dato exrtraido ",predicted[0])
                cv2.circle(screen, (955, 540), 100, (255, 0, 0), -1)
            else:
                predictedEyeLeft = blink.predict(eyeLeft)
                predictedEyeRight = blink.predict(eyeRight)
                image = cv2.putText(screen, 'Vista seteada', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(screen, 'Direccion: '+direccion, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



                if isOpen(predictedEyeLeft) == 'cerrado' and isOpen(predictedEyeRight) == 'cerrado' and direccion != "":
                    tCerradoFin = time.time()
                    if (tCerradoInicio-tCerradoFin < 2):
                        cv2.putText(screen, 'Seleccionado: ' + direccion, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("window", screen)
                        cv2.waitKey()
                        break
                    tCerradoInicio = time.time()
                else:
                    if abs(predicted[0][0] - np.mean(promedioX)) <= 3 and abs(
                            predicted[0][1] - np.mean(promedioY)) <= 1:
                        direccion = "centro"
                        cv2.circle(screen, (960, 540), 100, (255, 0, 0), -1)
                    else:
                        xPredict = predicted[0][0]
                        yPredict = predicted[0][1]
                        xPredict = predicted[0][0] - np.mean(promedioX)
                        yPredict = predicted[0][1] - np.mean(promedioY)
                        if (yPredict > 1 and xPredict < 3 and xPredict > -3):
                            direccion = "arriba"
                            # print("izquierda inferior")
                            cv2.circle(screen, (960, 0), 100, (255, 0, 0), -1)
                        elif (yPredict < -1 and xPredict < 3 and xPredict > -3):
                            direccion = "abajo"
                            # print("derecha inferior")
                            cv2.circle(screen, (960, 1080), 100, (255, 0, 0), -1)
                        elif (xPredict > 3):
                            direccion = "derecha"
                            # print("derecha superior")
                            cv2.circle(screen, (1920, 540), 100, (255, 0, 0), -1)
                        elif (xPredict < -3):
                            direccion = "izquierda"
                            # print("izquierda superior")
                            cv2.circle(screen, (0, 540), 100, (255, 0, 0), -1)

            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
