import os
import cv2
from tqdm import tqdm

image_folder = '../data/image'  # path to folder containing images
for file in tqdm(os.listdir(image_folder)):
    image_file = os.path.join(image_folder, file)
    cv2.imread(image_file)
