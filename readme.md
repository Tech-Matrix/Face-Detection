# Documentation

<p align="center">
<img width=300px height=200px src="https://cdn.wccftech.com/wp-content/uploads/2017/09/170531080448-jetblue-facial-recognition-1100x619.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Face Detection</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/Tech-Matrix/Face-Detection/pulls?q=is%3Aopen+is%3Apr)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 
  Face detection is an extremely important technology that is used widely around the world and has many uses such as combing cctv footage detecting individuals which are used for large scale purposes as well as small scale purposes like unlocking phones
</p>

## 📝 Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Built Using](#built_using)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

The aim for this project is to detect the individual's face using a box shape image as well as other parts of the face.

Other aims include emotion detection and object identification in the case that there is no presence of a face.
This repository is written in python and uses the opencv library as it already has tools which can be used to record a live video as well as have a method to mark facial features with certain shapes.
We utilise HaarCascades so as to avoid having to write a separate object detection algorithm, however their scope is limited due to limited number of haarCascades.

## 🏁 Getting Started <a name = "getting_started"></a>

### 📃 Prerequisites <a name="prerequisites"></a>

What things you need to install the software and how to install them.

- Install OpenCV
- Install Python 3.7 (For best use as it is configured to OpenCV)
- Atleast one webcam(inbuilt/Added)

### 🎈 Usage <a name="usage"></a>

1. Download the right python package(3.7).
1. Be careful when downloading the correct haarCascades.
1. Execute the Python script.
1. Note that you have specified your webcam in the script and are not using it for other purposes. [ cv2.VideoCapture(0) ]
1. Also note the exit button needed to close the program. [ Use cv2.waitkey() ]

### ⛏️ Built Using <a name = "built_using"></a>

- [Python](https://www.python.org/downloads/release/python-370/) - Language
- [OpenCV](https://sourceforge.net/projects/opencvlibrary/) - Additional Library
- [HaarCascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) - HaarCascades

### 🎉 Acknowledgements <a name = "acknowledgement"></a>

- [Tech with Tim](https://www.youtube.com/c/TechWithTim)
- [OpenCV-Documentation](https://docs.opencv.org/4.5.3/)
- [Net Ninjas](https://www.youtube.com/c/TheNetNinja)

