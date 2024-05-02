from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations

def convertBack(x, y, w, h):
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    global currentframe
    currentframe += 1
    if len(detections) > 0:
        persons = dict()
        handguns = dict()
        faces = dict()
        objectId = 0
        for detections in detections:
            name_tag = detections[0]
            print(name_tag)

            if name_tag == 'Person':
                x,y,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(x),float(y),float(w),float(h))
                persons[objectId] = (float(x), float(y), xmin, ymin, xmax, ymax)
                objectId += 1
                print("xmin = " + " " + str(xmin) + " " + "xmax = " + " " + str(xmax) + " " + "ymin = " + " " +
                      str(ymin) + " " + "ymax = " + " " + str(ymax))

            elif name_tag == 'Handgun':
                x2,y2,w2,h2 = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin2, ymin2, xmax2, ymax2 = convertBack(float(x2),float(y2),float(w2),float(h2))
                handguns[objectId] = (float(x2), float(y2), xmin2, ymin2, xmax2, ymax2)
                objectId += 1
                print("xmin2 = " + " " + str(xmin2) + " " + "xmax2 = " + " " + str(xmax2) + " " + "ymin2 = " + " " +
                      str(ymin2) + " " + "ymax2 = " + " " + str(ymax2))

            elif name_tag == 'Face':
                x3,y3,w3,h3 = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin3, ymin3, xmax3, ymax3 = convertBack(float(x3),float(y3),float(w3),float(h3))
                faces[objectId] = (float(x3), float(y3), xmin3, ymin3, xmax3, ymax3)
                objectId += 1
                print("xmin3 = " + " " + str(xmin3) + " " + "xmax3 = " + " " + str(xmax3) + " " + "ymin3 = " + " " +
                      str(ymin3) + " " + "ymax3 = " + " " + str(ymax3))

        #for box in persons.values():
         #   cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0,255,0), 2)

        archivo = open("./label/results" + "_video_test" + ".txt", "a")
        for bp in handguns.values():
            bp_xmid = bp[0]
            bp_ymid = bp[1]
            cv2.rectangle(img, (bp[2], bp[3]), (bp[4], bp[5]), (255, 0, 0), 2) #para cambiar de color si es un arma tirada o la tiene una pers.
            for per in persons.values():
                per_xmin = per[2]
                per_ymin = per[3]
                per_xmax = per[4]
                per_ymax = per[5]
                cv2.rectangle(img, (per_xmin, per_ymin), (per_xmax, per_ymax), (0, 0, 255), 2)

                if (per_xmin < bp_xmid and bp_xmid < per_xmax and
                        per_ymin < bp_ymid and bp_ymid < per_ymax):
                    ypredic_per = 1

                    print("object match")
                    cv2.rectangle(img, (per_xmin,per_ymin),(per_xmax,per_ymax), (255,255,255),2)
                    cv2.rectangle(img, (bp[2], bp[3]), (bp[4], bp[5]), (255, 0, 0), 2)#para cambiar de color si es un arma tirada o la tiene una pers.


                    for fac in faces.values():
                        fac_xmid = fac[0]
                        fac_ymid = fac[1]
                        fac_xmin = fac[2]
                        fac_ymin = fac[3]
                        fac_xmax = fac[4]
                        fac_ymax = fac[5]
                        if (per_xmin < fac_xmid and fac_xmid < per_xmax and
                                per_ymin < fac_ymid and fac_ymid < per_ymax):
                            print("object match")
                            cv2.rectangle(img, (fac_xmin, fac_ymin), (fac_xmax, fac_ymax), (255, 255, 0), 2)


                            #global currentframe
                            #currentframe += 1
                            x = fac[2]
                            y = fac[3]
                            h = (fac[5] - fac[3])
                            w = (fac[4] - fac[2])
                            print(h,w)
                            cropped_image = img[y:y + h, x:x + w]
                            try:
                                cv2.imshow("Cropped Image", cropped_image)  # Muestra la img cortada.
                                cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
                            except:
                                print(f"Error en frame {currentframe}")

                else:
                    ypredic_per = 0
                archivo.write(f"{currentframe},{per_xmin},{ypredic_per}\n")

    return img



netMain = None
metaMain = None
altNames = None
currentframe=-1 #Creamos una variable global q usamos en el contador de cropped faces.

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
    cap = cv2.VideoCapture("./videos_entrada/video_test.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Resolution: ",(width, height))

    out = cv2.VideoWriter(
            "./videos_salida/center.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
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
        print("FPS: " + str(1/(time.time()-prev_time)))
        cv2.imshow('Demo', image)
        #cv2.waitKey(3)
        if cv2.waitKey(1) == ord("q"):
            break
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
