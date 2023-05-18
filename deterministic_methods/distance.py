import math
import os
import time

import cv2

import darknet


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
                xmin = round((xmin / 960), 6)
                ymin = round((ymin / 540), 6)
                xmax = round((xmax / 960), 6)
                ymax = round((ymax / 540), 6)
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
                xmin2 = round((xmin2 / 960), 6)
                ymin2 = round((ymin2 / 540), 6)
                xmax2 = round((xmax2 / 960), 6)
                ymax2 = round((ymax2 / 540), 6)
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
                xmin3 = round((xmin3 / 960), 6)
                ymin3 = round((ymin3 / 540), 6)
                xmax3 = round((xmax3 / 960), 6)
                ymax3 = round((ymax3 / 540), 6)
                print("xmin3 = " + " " + str(xmin3) + " " + "xmax3 = " + " " + str(xmax3) + " " + "ymin3 = " + " " +
                      str(ymin3) + " " + "ymax3 = " + " " + str(ymax3))

        for fac in faces.values():
            fac_xmid = fac[0]
            fac_ymid = fac[1]
            fac_xmin = fac[2]
            fac_ymin = fac[3]
            fac_xmax = fac[4]
            fac_ymax = fac[5]
            min_dist = 10000
            min_per_xmid, min_per_ymid = 0, 0
            min_per_xmin, min_per_ymin, min_per_xmax, min_per_ymax = 0, 0, 0, 0
            for per in persons.values():
                per_xmid = per[0]
                per_ymid = per[1]
                per_xmin = per[2]
                per_ymin = per[3]
                per_xmax = per[4]
                per_ymax = per[5]
                p1 = fac_xmid - per_xmid
                p2 = fac_ymid - per_ymid
                dist = math.sqrt(p1 ** 2 + p2 ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_per_xmid = per_xmid
                    min_per_ymid = per_ymid
                    min_per_xmin = per_xmin
                    min_per_xmax = per_xmax
                    min_per_ymin = per_ymin
                    min_per_ymax = per_ymax
            print(min_per_xmin, min_per_ymin, min_per_xmax, min_per_ymax, min_dist)
            print(f"Distancia minima fac: {min_dist}")
            cv2.rectangle(img, (fac_xmin, fac_ymin), (fac_xmax, fac_ymax), (255, 255, 0), 2)
            cv2.rectangle(img, (min_per_xmin, min_per_ymin), (min_per_xmax, min_per_ymax), (0, 0, 255), 2)
            cv2.line(img, (int(fac_xmid), int(fac_ymid)), (int(min_per_xmid), int(min_per_ymid)), (0, 0, 255), 2)

        archivo = open("./label/results" + "_video_test" + ".txt", "a")
        for hg in handguns.values():
            hg_xmid = hg[0]
            hg_ymid = hg[1]
            hg_xmin = hg[2]
            hg_ymin = hg[3]
            hg_xmax = hg[4]
            hg_ymax = hg[5]
            min_dist = 10000
            min_per_xmid, min_per_ymid = 0, 0
            min_per_xmin, min_per_ymin, min_per_xmax, min_per_ymax = 0, 0, 0, 0
            distances = []
            for per in persons.values():
                per_xmid = per[0]
                per_ymid = per[1]
                per_xmin = per[2]
                per_ymin = per[3]
                per_xmax = per[4]
                per_ymax = per[5]
                p1 = hg_xmid - per_xmid
                p2 = hg_ymid - per_ymid
                dist = math.sqrt(p1 ** 2 + p2 ** 2)
                print(f"distancia hg: {dist}")

                distances.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    min_per_xmid = per_xmid
                    min_per_ymid = per_ymid
                    min_per_xmin = per_xmin
                    min_per_xmax = per_xmax
                    min_per_ymin = per_ymin
                    min_per_ymax = per_ymax

            for per, dista in enumerate(distances):
                if dista > min_dist:
                    ypredic_per = 0
                    archivo.write(f"{currentframe},{per},{per_xmin},{ypredic_per}\n")
                else:
                    ypredic_per = 1
                    archivo.write(f"{currentframe},{per},{min_per_xmin},{ypredic_per}\n")
                    #cv2.rectangle(img, (min_per_xmin, min_per_ymin), (min_per_xmax, min_per_ymax), (255, 255, 255), 2)


            print(min_per_xmin, min_per_ymin, min_per_xmax, min_per_ymax, min_dist)
            print(f"Distancia minima hg: {min_dist}")
            cv2.rectangle(img, (hg_xmin, hg_ymin), (hg_xmax, hg_ymax), (255, 0, 0), 2)
            cv2.rectangle(img, (min_per_xmin, min_per_ymin), (min_per_xmax, min_per_ymax), (255, 255, 255), 2)
            cv2.line(img, (int(hg_xmid), int(hg_ymid)), (int(min_per_xmid), int(min_per_ymid)), (255, 255, 255), 2)


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
    cap = cv2.VideoCapture("./videos_entrada/trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Resolution: ",(width, height))

    out = cv2.VideoWriter(
            "./videos_salida/distance.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
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
        #cv2.waitKey(3)
        if cv2.waitKey(1) == ord("q"):
            break
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()