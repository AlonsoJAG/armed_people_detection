# Armed People Detection
The proposed solution involves the development of algorithms for identifying people carrying handguns (pistols and revolvers). We have chosen the YOLOv4 model to detect people, guns,
and faces. Then, we extract information from YOLO related to real-time videos, such as bounding box coordinates, distances, and intersection areas between firearms and the people in each video frame to recognize the armed people.
## Table of Contents

* [Armed People Detection Algorithm](#Description-algorithm)

* [Dataset Generator](#Generator-Description)

* [Datasets](#Dataset)

* [Heuristics](#Heuristc-description)

* [Videos](#Videos)

* [Yolo](#Yolo)

* [Conclusión](#conclusión)

## Armed People Detection Algorithm
The algorithm is used to detect armed people. The file is saved in the armed_people_detection_algorithm folder with the name accuracy. This algorithm works together with YOLOv4 so the darknet module must be imported. The armed person and face detection models have been trained in Jupyter notebook and imported into this algorithm through the use of the pickle library. These trained models are shared in the folder named models/ml_armed_people_detection and models/ml_faces_armed_people_detection.
