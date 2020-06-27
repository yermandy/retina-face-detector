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

To exctract bounding boxes and landmarks use

``` Shell
python detection.py --ref casia_files.csv
```