import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm


def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image_1, image_2, model_tokenizer):
    start = time.process_time()
    image_1 = Image.open(image_1).convert("RGB")  # converting to 3 channels
    image_1 = np.array(image_1) / 255
    if image_2 is None:
        image_2 = image_1
    else:
        image_2 = Image.open(image_2).convert("RGB")  # converting to 3 channels
        image_2 = np.array(image_2) / 255
    st.image([image_1, image_2], width=300)
    caption = cm.function1([image_1], [image_2], model_tokenizer)
    st.markdown(" ### **Impression:**")
    impression = st.empty()
    impression.write(caption[0])
    print(" ### **Impression:**")
    print(caption[0])
    time_taken = "Time Taken for prediction: %i seconds" % (time.process_time() - start)
    print(time_taken)
    st.write(time_taken)
    del image_1, image_2


def predict_sample(model_tokenizer, folder='./test_images'):
    file_path = os.path.join(folder, str(1))
    if len(os.listdir(file_path)) == 2:
        image_1 = os.path.join(file_path, os.listdir(file_path)[0])
        image_2 = os.path.join(file_path, os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path, os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1, image_2, model_tokenizer)


model_tokenizer = create_model()
predict_sample(model_tokenizer)
