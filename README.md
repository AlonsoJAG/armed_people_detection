<h1 align="center"> Armed People Detection </h1> 

<p align="center">
  <img width="460" height="300" src="[https://picsum.photos/460/300](https://github.com/user-attachments/assets/c7ac8d79-0978-481b-baa9-2555236af611)">
</p>

![Imagen1](https://github.com/user-attachments/assets/c7ac8d79-0978-481b-baa9-2555236af611)

The proposed solution involves the development of algorithms for identifying people carrying handguns (pistols and revolvers). We have chosen the YOLOv4 model to detect people, guns,
and faces. Then, we extract information from YOLO related to real-time videos, such as bounding box coordinates, distances, and intersection areas between firearms and the people in each video frame to recognize the armed people.
## Table of Contents

* [Armed People Detection Algorithm](armed_people_detection_algorithm)

* [Dataset Generator](dataset_generator)

* [Datasets](datasets)

* [Heuristics](heuristics)

* [Videos](videos)

* [Yolo](yolo)

## Armed People Detection Algorithm
The algorithm is used to detect armed people. The file is saved in the armed_people_detection_algorithm folder with the name accuracy. This algorithm works together with YOLOv4 so the [darknet module](https://github.com/AlexeyAB/darknet) must be imported. The armed person and face detection models have been trained in [Jupyter Notebook](notebooks) and imported into this algorithm through the use of the pickle library. These trained models are shared in the folder named [models/ml_armed_people_detection](models/ml_armed_people_detection) and [models/ml_faces_armed_people_detection](models/ml_faces_armed_people_detection).

The dataset used for the MLP, KNN, and SVM training process was standardized before training using the function StandardScaler from the Scikit-Learn library (Jupyter Notebook). However, we import the models into our general system to receive the input data from YOLO’s live stream. It implies that the input data must be in the same conditions as the training process. Consequently, it was mandatory to standardize the input data in real-time, so we have applied the mathematical formula used by the StandardScaler function according to z = (x − u)/s, where x represents the input data to be standardized, u stands for the mean, and s is the standard deviation of the training samples. Below is the mean and standard deviation used to normalize the data received by YOLO in real time.

```
Line 9:
    To normalize the MLP, KNN, SVM data (All predictors-training dataset - 28 predictors):
    
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
```
```
On line 189, we can unlock the normalization of the data:
Line 189:
    #predictors_per = ((predictors_per - u)) / s # Only for mlp, knn, and svm. <--------------
    predictors_per = np.array(predictors_per).reshape(1,-1)
    print(predictors_per)
    ypredic_per = loaded_model_per.predict(predictors_per)
    print(f"Prediction_person {nper}: {ypredic_per}")
```

```
On line 322, we can find the code to select the models to use both to detect people and their faces:
Line 322:
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
```
On line 377, we can unlock the code that allows us to make detections through the web camera in real-time. On line 378, we can specify the path where the video we want to work with is.
```
Line 377:
    #cap = cv2.VideoCapture(0)
Line 378:
    cap = cv2.VideoCapture("./videos_entrada/trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
```
On line 93, we can modify the position where the file with the results of the processed video will be saved. This shows us that people were armed and unarmed. In lines 202 and 208 we can modify the predictors that we want to be shown in the results.
```
Line 93:
    archivo = open("./label/results" + "_video_test" + ".txt", "a")
```
```
Lines 202 and 208:
    archivo.write(f"{currentframe},{nper},{per_xmid},{ypredic_per},{prob0},{prob1},{prediction}\n")
```
On line 309, you can modify the route where the faces of armed people are stored.
```
Line 309:
    cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
```

## Dataset Generator
The [dataset generator](dataset_generator/dataset_generator.py) file is used to generate the dataset used in training and testing. On line 212 of the code we can choose the video from which we will extract the data to generate the dataset. On line 211 we can unlock the code that allows us to generate datasets in real time through the web camera.
```
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
```
In line 218 we can modify the location where the video will be saved with the detections generated by YOLO.
```
    out = cv2.VideoWriter(
            "./trasera_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
```
## Dataset 
Two files have been shared in the [dataset folder](datasets/armed_people_detection). The file named Videos1_2_3.csv was used to train the armed people detection models. The training dataset was generated by processing three videos. The video_test.csv file was used as the test dataset. These datasets have been developed with the [videos](videos) that are shared in the repository. These have been processed by the [dataset generator](dataset_generator) algorithm. The ground truth was placed manually for each record of the dataset, verifying the possession of the weapons in each frame of the video.

## Heuristics
The heuristics we propose are: Deterministic Method of Centers (DMC), Deterministic Method of Distances (DMD), and Deterministic Method of Intersections (DMI). It is possible to change the path where the results of the processed videos are saved. These are located on the following lines of code, DMI on line 177, DMC on line 90, and DMD on line 129.
```
Lines 177 for DMI, 90 for DMC, and 129 for DMD: 
    archivo = open("./label/results" + "_video_test" + ".txt", "a")
```
For heuristics, it is also possible to choose between processing videos or using the webcam in real-time. These are located on the following lines of code, DMI on line 320, DMC on line 193, and DMD on line 229.
```
Lines 320 for DMI, 193 for DMC, and 229 for DMD:
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./videos_entrada/trasera.mp4")   # <----- Replace with your video directory
```
The processed videos are saved automatically. The path where they are stored can be edited in the following lines of code: DMI on line 328, DMC on line 201, and DMD on line 237.
```
    out = cv2.VideoWriter(
            "./videos_salida/distance.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
```
## YOLO
We used our dataset to train the YOLOv4 object detector. We trained it from scratch to recognize faces, handguns, and people in the video. We randomly divided our [dataset](datasets/yolo) into 4,000 images for training and 1,000 for testing.  Afterward, we downloaded YOLOv4 from [Alexey Bochkovskiy’s GitHub](https://github.com/AlexeyAB/darknet) (YOLOv4 creator’s GitHub repository). This repository explains in detail how to configure YOLO. In the folder that contains the YOLO dataset we can find four files, which detail the classes, the location of the training and test images. These files are necessary for its operation. Furthermore, we trained YOLOv4 for 6,000 iterations. The [YOLO folder](yolo) includes two files, one is the settings used in YOLO, and the other contains the training weights. These files will allow you to apply YOLO to detect the three classes: weapons, faces, and people.

