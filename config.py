__author__ = 'Rio'

from yacs.config import CfgNode as CN

config = CN()

""" ============== Path Config ================= """
# 服务器path
config.chexnet_weights = '../model'
config.data_folder = '../pickle_files'
config.glove_path = '../glove.6B.300d.txt'
config.modelPng_save = './model2.png'
config.modelSave_path = '../Medical_image_Reporting'
config.image_folder = '../data/image'
config.reports_folder = "../data/report"
config.finalPkl_ph = '../data/pickle_files'


# # local path
# config.chexnet_weights = 'D:/RIO/Code/data/chexnet/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
# config.data_folder = '../data/pickle_files'
# config.glove_path = 'D:/RIO/Code/data/glove/glove.6B.300d.txt'
# config.modelPng_save = './model2.png'
# config.modelSave_path = 'D:\RIO\Code\data\model'
# config.image_folder = '../data/image'
# config.finalPkl_ph = '../data/pickle_files'
# config.reports_folder = "../data/ecgen-radiology"

# hyper parameter
config.embedding_dim = 300
config.dense_dim = 512
config.lstm_units = 512  # =dense_dim
config.dropout_rate = 0.2
config.input_size = (224, 224)
config.lr = 10 ** -3  # 学习率设定
config.batch_size = 200
config.dropout_rate = 0.2