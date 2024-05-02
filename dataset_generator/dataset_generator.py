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
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(255,255,0),(255,0,255),(255,255,0)]

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
    global colors
    #currentframe +=1
    if len(detections) > 0:
        persons = dict()
        handguns = dict()
        perId, hgId = 0,0
        for detections in detections:
            name_tag = detections[0]
            #print(name_tag)

            if name_tag == 'Person':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                persons[perId] = (xmid, ymid, xmin, ymin, xmax, ymax)


                #print("xmin = " + " " + str(xmin) + " " + "xmax = " + " " + str(xmax) + " " + "ymin = " + " " +
                      #str(ymin) + " " + "ymax = " + " " + str(ymax))
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[perId+1], 2)
                perId += 1

            elif name_tag == 'Handgun':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                handguns[hgId] = (float(xmid), float(ymid), xmin, ymin, xmax, ymax)


                #print("name_tag = " + str(name_tag)  + " " + "xmin = " + " " + str(xmin) + " " + "xmax = " + " " + str(xmax) + " " + "ymin = " + " " +
                      #str(ymin) + " " + "ymax = " + " " + str(ymax))
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[hgId], 2)
                hgId += 1
        global currentframe
        currentframe += 1
        archivo = open("./label/frames" + "_trasera" + ".txt", "a")
        #archivo.write(f"currentframe,nper,per_xmid,per_ymid,per_xmin,per_ymin,per_xmax,per_ymax,nhg,hg_xmid,hg_ymid,hg_xmin,hg_ymin,hg_xmax,hg_ymax,interseccion,included_center,areai,areah,dist\n")
        for nper, per in enumerate(persons.values()):
            per_xmid = per[0]
            per_ymid = per[1]
            per_xmin = per[2]
            per_ymin = per[3]
            per_xmax = per[4]
            per_ymax = per[5]
            for nhg, hg in enumerate(handguns.values()):
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
                if (per_xmin < hg_xmid and hg_xmid < per_xmax and per_ymin < hg_ymid and hg_ymid < per_ymax):
                    included_center = 1
                else:
                    included_center = 0

                if hg_xmax < per_xmin or hg_ymax < per_ymin or hg_xmin > per_xmax or hg_ymin > per_ymax:
                    interseccion = "No_intersection"
                    pass
                elif hg_xmin >= per_xmin and hg_xmax <= per_xmax and hg_ymin <= per_ymin and hg_ymax >= per_ymin:  # SC
                    ai = hg_xmax - hg_xmin
                    bi = hg_ymax - per_ymin
                    interseccion = "Up_center"

                elif hg_xmax >= per_xmin and hg_xmin <= per_xmin and hg_ymax >= per_ymin and hg_ymin <= per_ymin:  # SI
                    ai = hg_xmax - per_xmin
                    bi = hg_ymax - per_ymin
                    interseccion = "Up_left"

                elif hg_xmax >= per_xmax and hg_xmin <= per_xmax and hg_ymax >= per_ymin and hg_ymin <= per_ymin:  # SD
                    ai = per_xmax - hg_xmin
                    bi = hg_ymax - per_ymin
                    interseccion = "Up_right"

                elif hg_xmax >= per_xmin and hg_xmin <= per_xmin and hg_ymax >= per_ymax and hg_ymin <= per_ymax:  # II
                    ai = hg_xmax - per_xmin
                    bi = per_ymax - hg_ymin
                    interseccion = "Down_left"

                elif hg_xmax >= per_xmin and hg_xmin >= per_xmin and hg_ymax >= per_ymax and hg_ymin <= per_ymax:  # IC
                    ai = hg_xmax - hg_xmin
                    bi = per_ymax - hg_ymin
                    interseccion = "Down_center"

                elif hg_xmax >= per_xmax and hg_xmin <= per_xmax and hg_ymax >= per_ymax and hg_ymin <= per_ymax:  # ID
                    ai = per_xmax - hg_xmin
                    bi = per_ymax - hg_ymin
                    interseccion = "Down_right"

                elif hg_xmax >= per_xmax and hg_xmin <= per_xmax and hg_ymax <= per_ymax and hg_ymin >= per_ymin:  # CD
                    ai = per_xmax - hg_xmin
                    bi = hg_ymax - hg_ymin
                    interseccion = "Center_right"

                elif hg_xmax >= per_xmin and hg_xmin <= per_xmin and hg_ymax <= per_ymax and hg_ymin >= per_ymin:  # CI
                    ai = hg_xmax - per_xmin
                    bi = hg_ymax - hg_ymin
                    interseccion = "Center_left"

                elif per_xmin < hg_xmid and hg_xmid < per_xmax and per_ymin < hg_ymid and hg_ymid < per_ymax:  # Hg_in_per
                    ai = hg_xmax - hg_xmin
                    bi = hg_ymax - hg_ymin
                    interseccion = "Inside"

                areai = ai * bi

                archivo.write(f"{currentframe},{nper},{per_xmid},{per_ymid},{per_xmin},{per_ymin},{per_xmax},{per_ymax},{nhg},{hg_xmid},{hg_ymid},{hg_xmin},{hg_ymin},{hg_xmax},{hg_ymax},{interseccion},{included_center},{areai},{areah},{dist}\n")
        archivo.close()
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
    cap = cv2.VideoCapture("./trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Resolution: ",(width, height))

    out = cv2.VideoWriter(
            "./trasera_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
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
        #print("\n")
        #print(f"FRAME: {frame}")
        #print("FPS: " + str(1/(time.time()-prev_time)))
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
