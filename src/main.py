from detection.detect import detect_plate
from detection.segment_characters import segment_characters
from recognition.recognize import recognize_chracters
import argparse
import matplotlib.pyplot as plt
import os 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--i', '-image', help="licence plate to read", type=str)
args = parser.parse_args()



if args.i:
    image_path = args.i 
else:
    image_path = 'number-plate3.jpg'


t_value = []
t_name = []
templates = {}
templates_path = 'templates-blueprint/'
for template_name in os.listdir(templates_path):
    template = cv2.imread(templates_path + template_name, cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(template, (28,28))
    template[template < 15] = 0
    template[template > 15] = 1
    t_name.append(template_name.split(".")[0])
    t_value.append(template)

detected_plate = detect_plate(image_path)
segmented_characters = segment_characters(detected_plate, image_path)

if len(segmented_characters) > 10:
    recognized_characters= recognize_chracters(segmented_characters[1:], t_name, t_value)
else:
    recognized_characters= recognize_chracters(segmented_characters, t_name, t_value)

print("#"*50)
print(" ".join(recognized_characters))
print("*"*50)