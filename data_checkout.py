import os
import cv2
from tqdm import tqdm

# image_folder = '../data/image'  # path to folder containing images
# for file in tqdm(os.listdir(image_folder)):
#
#     image_file = os.path.join(image_folder, file)
#     img = cv2.imread(image_file)
#     if img is None:
#         print('None:', image_file)

# img1 = cv2.imread('../data/image/CXR4_IM-2050-2001.png')
img2 = cv2.imread('./CXR2185_IM-0795-2001.png')

# print(img1.shape)
print(img2.shape)