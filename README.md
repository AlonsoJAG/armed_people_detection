# Armed People Detection
The proposed solution involves the development of algorithms for identifying people carrying handguns (pistols and revolvers). We have chosen the YOLOv4 model to detect people, guns,
and faces. Then, we extract information from YOLO related to real-time videos, such as bounding box coordinates, distances, and intersection areas between firearms and the people in each video frame to recognize the armed people.
## Table of Contents

* [Armed People Detection Algorithm](armed_people_detection_algorithm)

* [Dataset Generator](#Generator-Description)

* [Datasets](#Dataset)

* [Heuristics](#Heuristc-description)

* [Videos](#Videos)

* [Yolo](#Yolo)

* [Conclusión](#conclusión)

## Armed People Detection Algorithm
The algorithm is used to detect armed people. The file is saved in the armed_people_detection_algorithm folder with the name accuracy. This algorithm works together with YOLOv4 so the [darknet module](https://github.com/AlexeyAB/darknet) must be imported. The armed person and face detection models have been trained in Jupyter notebook and imported into this algorithm through the use of the pickle library. These trained models are shared in the folder named [models/ml_armed_people_detection](models/ml_armed_people_detection) and [models/ml_faces_armed_people_detection](models/ml_faces_armed_people_detection).

The dataset used for the MLP, KNN, and SVM training process was standardized before training using the function StandardScaler from the Scikit-Learn library (Jupyter Notebook). However, we import the models into our general system to receive the input data from YOLO’s live stream. It implies that the input data must be in the same conditions as the training process. Consequently, it was mandatory to standardize the input data in real-time, so we have applied the mathematical formula used by the StandardScaler function according to z = (x − u)/s, where x represents the input data to be standardized, u stands for the mean, and s is the standard deviation of the training samples. Below is the mean and standard deviation used to normalize the data received by YOLO in real time.



```
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
On line 189 we can unlock the normalization of the data:

                #predictors_per = ((predictors_per - u)) / s # Only for mlp, knn, and svm. <--------------
                predictors_per = np.array(predictors_per).reshape(1,-1)
                print(predictors_per)
                ypredic_per = loaded_model_per.predict(predictors_per)
                print(f"Prediction_person {nper}: {ypredic_per}")
```

```
On line 322 we can find the code to select the models to use both to detect people and their faces:
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
## Dataset Generator
The [dataset generator](dataset_generator/dataset_generator.py) file is used to generate the dataset used in training and testing. On line 212 of the code we can choose the video from which we will extract the data to generate the dataset. 

