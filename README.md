## RetinaFace detector

### Installation

Organise the dataset directory as follows:

``` Shell
./
  resources/
    casia_files.csv
  images/
    casia/
  weights/
    Resnet50_Final.pth
  
```

Use the [following link](https://drive.google.com/drive/folders/1Sb68JNN5hGnjOS65SgzTCE7QQoi1xaRq?usp=sharing) to download data necessary for feature extraction. Place CASIA dataset to ```casia``` folder.

To exctract bounding boxes and landmarks use the following script:

``` Shell
python detection.py --ref casia_files.csv
```