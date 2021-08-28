import os
import cv2
from tqdm import tqdm
from config import config as args


image_folder = args.image_folder  # path to folder containing images
for file in tqdm(os.listdir(image_folder)):

    image_file = os.path.join(image_folder, file)
    img = cv2.imread(image_file)
    if img is None:
        print('None:', image_file)
