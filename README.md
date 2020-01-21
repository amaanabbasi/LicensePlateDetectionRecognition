# LicensePlateDetectionRecognition
Automating the task of recognition of license plate characters.

#Credits
[Achraf KHAZRI](https://towardsdatascience.com/automatic-license-plate-detection-recognition-using-deep-learning-624def07eaaf)

# Prerequisites
- Download the lapi.weights file and paste it into /src/detection.

# Installing dependencies (Python3)
- pip install -r requirements.txt

# Usage
python main.py --i=number-plate.jpg

# Directory Overview

- src/detection: contains the code related to the detection of the licenseplate in the scene,
                 extraction and segmentation of characters.

- src/recognition: Recognition code, currently using a basic algorithm. 

- src/results: contains the images ofthe extracted licenseplate and segmented characters.

