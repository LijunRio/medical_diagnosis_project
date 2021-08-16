__author__ = 'Rio'
from yacs.config import CfgNode as CN

config = CN()

""" ============== Path Config ================= """
# # 服务器path
# config.chexnet_weights = '../model/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
# config.data_folder = '../pickle_files'
# config.glove_path = '../glove.6B.300d.txt'
# config.modelPng_save = './model.png'
# config.modelSave_path = '../Medical_image_Reporting'

# local path
config.chexnet_weights = 'D:/Code/SIAT_2021/data/model/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
config.data_folder = 'D:/Code/SIAT_2021/data/pickle_files'
config.glove_path = 'D:/Code/SIAT_2021/data/glove.6B.300d.txt'
config.modelPng_save = './model2.png'
config.modelSave_path = 'D:\Code\SIAT_2021\data\Medical image Reporting'

# hyper parameter
config.embedding_dim = 300
config.dense_dim = 512
config.lstm_units = 512  # =dense_dim
config.dropout_rate = 0.2
config.input_size = (224, 224)
config.lr = 10 ** -3  # 学习率设定
config.batch_size = 100
