import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, LSTM, Dot, Reshape, Concatenate, \
    BatchNormalization, GlobalMaxPooling2D, Dropout, Add
import numpy as np
from DataLoader import tokenizing_analysis
import pandas as pd
import os
from DataLoader import Dataloader, Dataset


# chexnet weights ; https://drive.google.com/file/d/19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b/view
def create_chexnet(chexnet_weights):
    """
  chexnet_weights: weights value in .h5 format of chexnet
  creates a chexnet model with preloaded weights present in chexnet_weights file
  """
    # importing densenet the last layer will be a relu activation layer
    model = tf.keras.applications.DenseNet121(include_top=False)

    # we need to load the weights so setting the architecture of the model as same as the one of tha chexnet
    x = model.output  # output from chexnet
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)
    # here activation is sigmoid as seen in research paper

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[
        -2].output)  # we will be taking the penultimate layer (second last layer here it is global avgpooling)
    return chexnet


class Image_encoder(tf.keras.layers.Layer):
    """
    This layer will output image backbone features after passing it through chexnet
    here chexnet will be not be trainable
    """

    def __init__(self, name="image_encoder_block"):
        super().__init__()
        self.chexnet_weights = '../model/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
        self.chexnet = create_chexnet(self.chexnet_weights)
        self.chexnet.trainable = False
        for i in range(10):  # the last 10 layers of chexnet will be trained
            self.chexnet.layers[-i].trainable = True

    def call(self, data):
        op = self.chexnet(data)
        return op


embedding_dim = 300
dense_dim = 512
lstm_units = dense_dim
dropout_rate = 0.2

# load data
folder_name = '../pickle_files'
file_name = 'train.pkl'
train = pd.read_pickle(os.path.join(folder_name, file_name))
file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(folder_name, file_name))

# DataLoader Part
input_size = (224, 224)
tokenizer, max_pad, test_captions, vocab_size, start_index, end_index = tokenizing_analysis(train=train, test=test)
print("max_pad:", max_pad)

glove = {}  # glove用于将词向量化
with open('../glove.6B.300d.txt', encoding='utf-8') as f:  # taking 300 dimesions
    for line in f:
        word = line.split()  # it is stored as string like this "'the': '.418 0.24968 -0.41242 0.1217 0.34527
        # -0.044457 -0.4"
        glove[word[0]] = np.asarray(word[1:], dtype='float32')

embedding_dim = 300  # 嵌入向量维度为300
# create a weight matrix for words in training docs for embedding purpose
embedding_matrix = np.zeros(
    (vocab_size + 1, embedding_dim))  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

for word, i in tokenizer.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:  # if the word is found in glove vectors
        embedding_matrix[i] = embedding_vector[:embedding_dim]

tf.keras.backend.clear_session()  # 输入大小(224,224,3)
# https://www.w3resource.com/python-exercises/tuple/python-tuple-exercise-5.php
image1 = Input(shape=(input_size + (3,)))  # shape = 224,224,3
image2 = Input(shape=(input_size + (3,)))
caption = Input(shape=(max_pad,))

# 使用chexnet进行图片编码
img_encoder = Image_encoder()  # contains chexnet model which is set trainable  =  False
# img2_encoder = Image_encoder() #opshape: (?,1024)
bk_feat1 = img_encoder(image1)  # 对image1进行编码
# bk_dense = Dense(dense_dim,
#                  activation = 'relu',
#                  name = 'bk_dense'
#                   )
# bk_feat1 = bk_dense(bk_feat1) #dense for the first image op: (?,dense_dim)

bk_feat2 = img_encoder(image2)  # 对image2 进行编码
# bk_feat2 = bk_dense(bk_feat2) #dense for the 2nd image op: (?,dense_dim)

# 将image1 和 image2 的特征向量进行concat
# concatenating the backbone images op_shape: (?,1024)
bk_features_concat = Concatenate(axis=-1)([bk_feat1, bk_feat2])

# bk_features_concat = BatchNormalization()(bk_features_concat) #applying batch norm
# bk_features_concat = Dropout(dropout_rate)(bk_features_concat)
image_dense = Dense(dense_dim,
                    activation='relu',
                    name='Image_dense',
                    use_bias='False'
                    )

# 将concat到一起的向量再通过dense net
# final op from dense op_shape:
image_bkbone = image_dense(bk_features_concat)  # (?,dense_dim) this will be added as initial states to the lstm
image_dense_op = tf.keras.backend.expand_dims(image_bkbone, axis=1)  # op_shape: (?,1,dense_dim)

embedding = Embedding(input_dim=vocab_size + 1,
                      output_dim=embedding_dim,
                      input_length=max_pad,
                      mask_zero=True,
                      weights=[embedding_matrix],
                      name='embedding'
                      )
embed_op = embedding(caption)  # op_shape: (?,input_length,embedding_dim)

lstm_layer = LSTM(units=lstm_units,
                  return_sequences=True,
                  return_state=True
                  )
# op_shape = batch_size*input_length*lstm_units
lstm_op, lstm_h, lstm_c = lstm_layer(embed_op, initial_state=[image_bkbone, image_bkbone])

# lstm_op = BatchNormalization()(lstm_op)
# op_shape: (?,input_lenght,lstm_units/dense_dim) here lstm_dims=dense_dim
add = Add()([image_dense_op, lstm_op])

op_dense = Dense(vocab_size + 1,
                 activation='softmax',
                 name='output_dense'
                 )  # op: (?,input_length,vocab_size+1)

output = op_dense(add)
print('output:', output)

#  网络搭建完毕！
model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
print(model.summary())

# 保存模型结构图
model_png = './model.png'
tf.keras.utils.plot_model(model, to_file=model_png, show_shapes=True)


def custom_loss(y_true, y_pred):
    # getting mask value to not consider those words which are not present in the true caption
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    print('mask', mask)

    y_pred = y_pred + 10 ** -7  # to prevent loss becoming null

    # calculating the loss
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_ = loss_func(y_true, y_pred)
    print('y_true:', y_true)
    print('y_pred:', y_pred)

    # converting mask dtype to loss_ dtype
    mask = tf.cast(mask, dtype=loss_.dtype)

    # applying the mask to loss
    loss_ = loss_ * mask

    # returning mean over all the values
    return tf.reduce_mean(loss_)


#
#
lr = 10 ** -3  # 学习率设定
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # optimizer
# model.compile(optimizer=optimizer,loss=custom_loss,metrics= ['accuracy'])  # custom loss有问题
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

tf.keras.backend.clear_session()
tb_filename = 'Simple_Encoder_Decoder/'
tb_file = os.path.join('../Medical_image_Reporting', tb_filename)
model_filename = 'Simple_Encoder_Decoder.h5'
model_save = os.path.join('../Medical_image_Reporting', model_filename)
my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, verbose=2),
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save, save_best_only=True,
                                                   save_weights_only=True, verbose=2),
                tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=tb_file),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                                     min_lr=10 ** -7, verbose=2)]  # from keras documentation

# load data
folder_name = '../pickle_files'
file_name = 'train.pkl'
train = pd.read_pickle(os.path.join(folder_name, file_name))
file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(folder_name, file_name))

# DataLoader Part
input_size = (224, 224)
batch_size = 100
tokenizer, max_pad, *_ = tokenizing_analysis(train=train, test=test, visualising=True)
print("max_pad:", max_pad)
train_dataloader = Dataset(train, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
train_dataloader = Dataloader(train_dataloader, batch_size=batch_size)

test_dataloader = Dataset(test, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
test_dataloader = Dataloader(test_dataloader, batch_size=batch_size)

model.fit(train_dataloader,
          validation_data=test_dataloader,
          epochs=10,
          callbacks=my_callbacks
          )
print('ok')
