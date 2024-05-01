import math
import os
import time
import pickle
import cv2
import darknet
import numpy as np

#To normalize the mlp data (All predictors-training dataset - 28 predictors)
u = np.array([9.09106623e+02, 1.05738223e+00, 4.01556492e+02, 4.15881999e+02,
              3.15145035e+02, 3.00285205e+02, 4.87967948e+02, 5.31478794e+02,
              3.87211508e-01, 3.15303209e+02, 3.59825385e+02, 2.92086914e+02,
              3.39647453e+02, 3.38519504e+02, 3.80003317e+02, 3.83417641e-01,
              9.68300700e+02, 2.15281262e+03, 2.31289485e+02, 1.56180841e-01,
              6.82105596e-02, 5.53272210e-04, 1.58077774e-04, 1.58631046e-01,
              5.61650332e-01, 2.06291495e-02, 5.29560544e-03, 2.86911160e-02])

s = np.array([4.43693600e+02, 9.50245842e-01, 2.16075512e+02, 4.92883235e+01,
              2.16727143e+02, 8.71466305e+01, 2.25161619e+02, 2.14580602e+01,
              6.72153476e-01, 1.68249783e+02, 7.25382729e+01, 1.65957845e+02,
              7.71079606e+01, 1.71079386e+02, 6.88302982e+01, 4.86218627e-01,
              1.90893059e+03, 2.14921043e+03, 1.47367328e+02, 3.63026701e-01,
              2.52106880e-01, 2.35152312e-02, 1.25719046e-02, 3.65331682e-01,
              4.96184680e-01, 1.42139325e-01, 7.25779719e-02, 1.66936922e-01])

#To normalize the mlp data (Uncomplete predictors-test dataset - 25 predictors)
u2 = np.array([5.34001565e+02, 5.00790436e+02, 3.26736635e+02, 3.39182004e+02,
               1.36543240e+02, 6.62398869e+02, 5.16930030e+02, 4.82650473e+02,
               2.37284146e+02, 4.44251197e+02, 2.05940383e+02, 5.21049750e+02,
               2.68627910e+02, 6.99530516e-01, 4.30825639e+03, 5.77297156e+03,
               2.03288357e+02, 2.91079812e-01, 1.61189358e-01, 1.09546166e-02,
               3.30203443e-01, 1.61189358e-01, 1.56494523e-03, 3.91236307e-02,
               4.69483568e-03])

s2 = np.array([3.03511404e+02, 2.01963212e+02, 2.48928166e+01, 2.26404402e+02,
               5.50486989e+01, 2.12388524e+02, 2.14366986e+01, 1.02296829e+02,
               6.73773981e+01, 9.73261329e+01, 7.00034889e+01, 1.09763883e+02,
               6.86790951e+01, 4.58462183e-01, 4.26570072e+03, 4.93661746e+03,
               8.87213816e+01, 4.54260228e-01, 3.67705520e-01, 1.04089447e-01,
               4.70286220e-01, 3.67705520e-01, 3.95284223e-02, 1.93889072e-01,
               6.83578393e-02])

def convertBack(x, y, w, h):
#Converts center coordinates to rectangle coordinates
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax



def cvDrawBoxes(detections, img):
    global currentframe
    currentframe += 1
    if len(detections) > 0:
        persons = dict()
        handguns = dict()
        faces = dict()
        perId, hgId, facId = 0,0,0
        for detections in detections:
            name_tag = detections[0]


            if name_tag == 'Person':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                persons[perId] = (xmid, ymid, xmin, ymin, xmax, ymax)
                perId += 1

            elif name_tag == 'Handgun':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                handguns[hgId] = (float(xmid), float(ymid), xmin, ymin, xmax, ymax)
                hgId += 1

            elif name_tag == 'Face':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                faces[facId] = (float(xmid), float(ymid), xmin, ymin, xmax, ymax)
                facId += 1


        archivo = open("./label/results" + "_video_test" + ".txt", "a")
        umbral = 0.51
        for nper, per in enumerate(persons.values()):
            per_xmid = per[0]
            per_ymid = per[1]
            per_xmin = per[2]
            per_ymin = per[3]
            per_xmax = per[4]
            per_ymax = per[5]
            for nhg, hg in enumerate(handguns.values()):
                Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                hg_xmid = hg[0]
                hg_ymid = hg[1]
                hg_xmin = hg[2]
                hg_ymin = hg[3]
                hg_xmax = hg[4]
                hg_ymax = hg[5]
                a2 = hg_xmax - hg_xmin
                b2 = hg_ymax - hg_ymin
                areah = a2 * b2
                areai, ai, bi = 0, 0, 0
                p1 = hg_xmid - per_xmid
                p2 = hg_ymid - per_ymid
                dist = math.sqrt(p1 ** 2 + p2 ** 2)


                if per_xmin < hg_xmid and hg_xmid < per_xmax and per_ymin < hg_ymid and hg_ymid < per_ymax:
                    included_center = 1
                else:
                    included_center = 0

                if hg_xmax < per_xmin or hg_ymax < per_ymin or hg_xmin > per_xmax or hg_ymin > per_ymax:
                    Intersection_No_intersection = 1
                else:
                    Intersection_No_intersection = 0

                    if hg_xmin < per_xmin: #Left side
                        if hg_ymin < per_ymin:          # SI
                            ai = hg_xmax - per_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_left = 1
                        elif hg_ymax > per_ymax:        # II
                            ai = hg_xmax - per_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_left = 1
                        else:                           # CI
                            ai = hg_xmax - per_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Center_left = 1


                    elif hg_xmax > per_xmax: #Rigth side
                        if hg_ymin < per_ymin:          # SD
                            ai = per_xmax - hg_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_right = 1
                        elif hg_ymax > per_ymax:        # ID
                            ai = per_xmax - hg_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_rigth = 1
                        else:                           # CD
                            ai = per_xmax - hg_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Center_right = 1


                    elif hg_xmin > per_xmin and hg_xmax < per_xmax: #center
                        if hg_ymin < per_ymin:          # SC
                            ai = hg_xmax - hg_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_center = 1
                        elif hg_ymax > per_ymax:        # IC
                            ai = hg_xmax - hg_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_center = 1
                        else:                           # Hg_in_per
                            ai = hg_xmax - hg_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Inside = 1

                areai = ai * bi


                predictors_per = [currentframe, nper, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, nhg, hg_xmid, hg_ymid, hg_xmin, hg_ymin,
                   hg_xmax, hg_ymax, included_center, areai, areah, dist, Intersection_Center_left, Intersection_Center_right,
                   Intersection_Down_center, Intersection_Down_left, Intersection_Inside, Intersection_No_intersection, Intersection_Up_center,
                   Intersection_Up_left, Intersection_Up_right]


                print(f'predictor_per {predictors_per}')
                print(f'ci={Intersection_Center_left}, cd={Intersection_Center_right}, ic={Intersection_Down_center}, ii={Intersection_Down_left},'
                      f' hginper={Intersection_Inside}, ni={Intersection_No_intersection}, sc={Intersection_Up_center}, si={Intersection_Up_left},'
                     f' sd={Intersection_Up_right}')

                #predictors_per = ((predictors_per - u)) / s # Only for mlp, knn, and svm.
                predictors_per = np.array(predictors_per).reshape(1,-1)
                print(predictors_per)
                ypredic_per = loaded_model_per.predict(predictors_per)
                print(f"Prediction_person {nper}: {ypredic_per}")

                probability = loaded_model_per.predict_proba(predictors_per)

                prob1 = probability[0, 1]
                prob0 = probability[0, 0]

                if prob1 < umbral:
                    prediction = 0
                    archivo.write(f"{currentframe},{nper},{per_xmid},{ypredic_per},{prob0},{prob1},{prediction}\n")


                else:
                    cv2.rectangle(img, (int(per_xmin), int(per_ymin)), (int(per_xmax), int(per_ymax)), (255, 0, 0), 1)
                    prediction = 1
                    archivo.write(f"{currentframe},{nper},{per_xmid},{ypredic_per},{prob0},{prob1},{prediction}\n")

                    for nfac, fac in enumerate(faces.values()):
                        fac_xmid = fac[0]
                        fac_ymid = fac[1]
                        fac_xmin = fac[2]
                        fac_ymin = fac[3]
                        fac_xmax = fac[4]
                        fac_ymax = fac[5]
                        a2 = fac_xmax - fac_xmin
                        b2 = fac_ymax - fac_ymin
                        areaf = a2 * b2
                        areai, ai, bi = 0, 0, 0
                        p1 = fac_xmid - per_xmid
                        p2 = fac_ymid - per_ymid
                        dist = math.sqrt(p1 ** 2 + p2 ** 2)

                        Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                        Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                        Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                        if (per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax):
                            included_center = 1
                        else:
                            included_center = 0

                        if fac_xmax < per_xmin or fac_ymax < per_ymin or fac_xmin > per_xmax or fac_ymin > per_ymax:
                            Intersection_No_intersection = 1
                            continue
                        else:
                            Intersection_No_intersection = 0

                        if  fac_xmin >= per_xmin and fac_xmax <= per_xmax and fac_ymin <= per_ymin and fac_ymax >= per_ymin:  # SC
                            ai = fac_xmax - fac_xmin
                            bi = fac_ymax - per_ymin
                            Intersection_Up_center = 1

                        elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SI
                            ai = fac_xmax - per_xmin
                            bi = fac_ymax - per_ymin
                            Intersection_Up_left = 1

                        elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SD
                            ai = per_xmax - fac_xmin
                            bi = fac_ymax - per_ymin
                            Intersection_Up_right = 1

                        elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # II
                            ai = fac_xmax - per_xmin
                            bi = per_ymax - fac_ymin
                            Intersection_Down_left = 1

                        elif fac_xmax >= per_xmin and fac_xmin >= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # IC
                            ai = fac_xmax - fac_xmin
                            bi = per_ymax - fac_ymin
                            Intersection_Down_center = 1

                        elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # ID
                            ai = per_xmax - fac_xmin
                            bi = per_ymax - fac_ymin
                            Intersection_Down_right = 1

                        elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CD
                            ai = per_xmax - fac_xmin
                            bi = fac_ymax - fac_ymin
                            Intersection_Center_right = 1

                        elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CI
                            ai = fac_xmax - per_xmin
                            bi = fac_ymax - fac_ymin
                            Intersection_Center_left = 1

                        elif per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax:  # Hg_in_per
                            ai = fac_xmax - fac_xmin
                            bi = fac_ymax - fac_ymin
                            Intersection_Inside = 1

                        areai = ai * bi


                        predictors_fac = [currentframe, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, fac_xmid, fac_ymid, fac_xmin, fac_ymin,
                                          fac_xmax, fac_ymax, included_center, areai, areaf, dist, Intersection_Center_left, Intersection_Center_right,
                                          Intersection_Inside, Intersection_No_intersection, Intersection_Up_center, Intersection_Up_left,
                                          Intersection_Up_right]

                        predictors_fac = np.array(predictors_fac).reshape(1,-1) # una fila y el resto de datos indistinto.

                        ypredic_fac = loaded_model_fac.predict(predictors_fac)

                        print(f"Prediction_face {nfac}: {ypredic_fac}")
                        if ypredic_fac == 1:
                            cv2.rectangle(img, (int(fac_xmin), int(fac_ymin)), (int(fac_xmax), int(fac_ymax)), (255, 0, 0), 1)

                            x = int(fac_xmin)
                            y = int(fac_ymin)
                            h = (int(fac_ymax) - int(fac_ymin))
                            w = (int(fac_xmax) - int(fac_xmin))
                            #print(h,w)
                            cropped_image = img[y:y + h, x:x + w]
                            try:
                                cv2.imshow("Cropped Image", cropped_image)  # Muestra la img cortada.
                                cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
                            except:
                                print(f"Error en frame {currentframe}")

        archivo.close()

    return img

netMain = None
metaMain = None
altNames = None
currentframe=-1 #Creamos una variable global q usamos en el contador de cropped faces.

#People detection model
#loaded_model_per = pickle.load(open('./modelos/hg/rfc_t4.sav', 'rb'))
loaded_model_per = pickle.load(open('./modelos/hg/rfc11.sav', 'rb'))
#loaded_model_per = pickle.load(open('./modelos/hg/mlp_t10.sav', 'rb'))
#loaded_model_per = pickle.load(open('./modelos/hg/mlp17.sav', 'rb'))
#loaded_model_per = pickle.load(open('./modelos/hg/bag_model.sav', 'rb'))

#Faces detection model
#loaded_model_fac = pickle.load(open('./modelos/fac/rfc_t.sav', 'rb'))
loaded_model_fac = pickle.load(open('./modelos/fac/rfc.sav', 'rb'))
#loaded_model_fac = pickle.load(open('./modelos/fac/mlp_t.sav', 'rb'))
#loaded_model_fac = pickle.load(open('./modelos/fac/mlp.sav', 'rb'))

def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4_2.cfg"
    weightPath = "./backup/yolov4_2_best2.weights"
    metaPath = "./data/custom/piford.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./videos_entrada/trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Resolution: ",(width, height))

    out = cv2.VideoWriter(
            "./videos_salida/prueba.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)

    frame = 0
    while True:
        frame += 1
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, namesList, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("\n")
        print(f"FRAME: {frame}")
        print("FPS: " + str(1/(time.time()-prev_time)))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        if cv2.waitKey(1) == ord("q"):
            break
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
