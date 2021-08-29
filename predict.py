import pandas as pd
import os
from config import config as args
from create_model import function1
import pandas

# read the test image path
file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(args.finalPkl_ph, file_name))
print(test.columns.values.tolist())
image1_pth = test['image_1'].values.tolist()
image2_pth = test['image_2'].values.tolist()

function1(image1_pth, image2_pth)
