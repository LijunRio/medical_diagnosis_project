import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm


def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict_sample(model_tokenizer, folder='./test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1, no_files)
    file_path = os.path.join(folder, str(file))
    if len(os.listdir(file_path)) == 2:
        image_1 = os.path.join(file_path, os.listdir(file_path)[0])
        image_2 = os.path.join(file_path, os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path, os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1, image_2, model_tokenizer, True)


model_tokenizer = create_model()
predict_sample(model_tokenizer)
