import pandas as pd
import os
from config import config as args
from create_model import function1, get_detail_result
from tqdm import tqdm
import cv2

# read the test image path
file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(args.finalPkl_ph, file_name))
print(test.columns.values.tolist())
image1_pth = test['image_1'].values.tolist()
image2_pth = test['image_2'].values.tolist()
impression_list = test['impression'].values.tolist()


# # checkout file
# image_folder = args.image_folder  # path to folder containing images
# for file1, file2 in tqdm(zip(image1_pth, image2_pth)):
#
#     img = cv2.imread(file1)
#     if img is None:
#         print('None:', file1)
#     img2 = cv2.imread(file2)
#     if img2 is None:
#         print('None:', file2)

# function1(image1_pth, image2_pth)

result = get_detail_result(image1_pth, image2_pth, impression_list)
print(len(result['image1'].values.tolist()))
result.to_csv('test_result.csv')
print('save result to csv file !')
