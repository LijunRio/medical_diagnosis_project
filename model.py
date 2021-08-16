import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, LSTM, Dot, Reshape, Concatenate, \
    BatchNormalization, GlobalMaxPooling2D, Dropout, Add
import numpy as np
from DataLoader import tokenizing_analysis
import pandas as pd
import os
from DataLoader import Dataloader, Dataset
from nltk.translate.bleu_score import sentence_bleu  # bleu score
import cv2
import matplotlib.pyplot as plt
from config import config as args
from tqdm import tqdm


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
        self.chexnet_weights = args.chexnet_weights
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
folder_name = args.data_folder
file_name = 'train.pkl'
train = pd.read_pickle(os.path.join(folder_name, file_name))
file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(folder_name, file_name))
test = test[:10]  # 仅仅用前10个做测试
print(test.shape)

# DataLoader Part
input_size = (224, 224)
tokenizer, max_pad, test_captions, vocab_size, start_index, end_index = tokenizing_analysis(train=train, test=test)
print("max_pad:", max_pad)

glove = {}  # glove用于将词向量化
with open(args.glove_path, encoding='utf-8') as f:  # taking 300 dimesions
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

tf.keras.backend.clear_session()  # 清除小节
# https://www.w3resource.com/python-exercises/tuple/python-tuple-exercise-5.php
image1 = Input(shape=(input_size + (3,)))  # shape = 224,224,3
image2 = Input(shape=(input_size + (3,)))
caption = Input(shape=(max_pad,))  # 第80百分位的长度，28

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
bk_features_concat = Concatenate(axis=-1)([bk_feat1, bk_feat2])  # (None, 2048)

# bk_features_concat = BatchNormalization()(bk_features_concat) #applying batch norm
# bk_features_concat = Dropout(dropout_rate)(bk_features_concat)
image_dense = Dense(dense_dim,  # 将2048压缩成512维
                    activation='relu',
                    name='Image_dense',
                    use_bias='False'
                    )

# 将concat到一起的向量再通过dense net
# final op from dense op_shape:

# 这里dense_dim = 512, 输出（None, 512）
image_bkbone = image_dense(bk_features_concat)  # (?,dense_dim) this will be added as initial states to the lstm
# 扩张一维
image_dense_op = tf.keras.backend.expand_dims(image_bkbone, axis=1)  # op_shape: (?,1,dense_dim)

# 嵌入层
embedding = Embedding(input_dim=vocab_size + 1,
                      output_dim=embedding_dim,
                      input_length=max_pad,
                      mask_zero=True,
                      weights=[embedding_matrix],  # 使用glove vector来初始化权重
                      name='embedding'
                      )

# （None, 28, 300）
embed_op = embedding(caption)  # op_shape: (?,input_length,embedding_dim)


# lstm_units = dense_dim = 512
# units：输出维度
# input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)
# return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
# input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，需要指定该参数，否则全连接的输出无法计算出来。

lstm_layer = LSTM(units=lstm_units, return_sequences=True, return_state=True)
# op_shape = batch_size*input_length*lstm_units

# 使用两张图片提取出来的特征向量作为LSTM的初始参数
# lstm_op (None, 28, 512), lstm_h(None, 512), lstm_c(None, 512)
lstm_op, lstm_h, lstm_c = lstm_layer(embed_op, initial_state=[image_bkbone, image_bkbone])

# lstm_op = BatchNormalization()(lstm_op)
# op_shape: (?,input_lenght,lstm_units/dense_dim) here lstm_dims=dense_dim
add = Add()([image_dense_op, lstm_op])  # (None, 28+1, 512)

op_dense = Dense(vocab_size + 1,  # 单词长度+1
                 activation='softmax',
                 name='output_dense'
                 )  # op: (?,input_length,vocab_size+1)

output = op_dense(add)
# print('output:', output)

#  网络搭建完毕！
model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
print(model.summary())

# 保存模型结构图
model_png = args.modelPng_save
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
tb_file = os.path.join(args.modelSave_path, tb_filename)
model_filename = 'Simple_Encoder_Decoder_0.h5'
model_save = os.path.join(args.modelSave_path, model_filename)
my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, verbose=2),
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save, save_best_only=True,
                                                   save_weights_only=True, verbose=2),
                tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=tb_file),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                                     min_lr=10 ** -7, verbose=2)]  # from keras documentation

# DataLoader Part
input_size = (224, 224)
batch_size = 100
tokenizer, max_pad, *_ = tokenizing_analysis(train=train, test=test)
print("max_pad:", max_pad)
train_dataloader = Dataset(train, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
train_dataloader = Dataloader(train_dataloader, batch_size=batch_size)

test_dataloader = Dataset(test, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
test_dataloader = Dataloader(test_dataloader, batch_size=batch_size)

# with tf.device("/device:GPU:0"):
#     model.fit(train_dataloader,
#               validation_data=test_dataloader,
#               epochs=10,
#               callbacks=my_callbacks
#               )

# predict part


model1 = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
print('model_save_path:', model_save)
model1.load_weights(model_save)


def get_bleu(reference, prediction):
    """
  Given a reference and prediction string, outputs the 1-gram,2-gram,3-gram and 4-gram bleu scores
  """
    reference = [reference.split()]  # should be in an array (cos of multiple references can be there here only 1)
    prediction = prediction.split()
    bleu1 = sentence_bleu(reference, prediction, weights=(1, 0, 0, 0), )
    bleu2 = sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


# calculate bleu scores for every datapoint
def mean_bleu(test, predict, model=model1, **kwargs):
    """
  given a df and predict fucntion which predicts the impression of the caption
  outpus the mean bleu1,bleu2,bleu3, bleu4 for entire datapoints in df
  """
    if kwargs != None:
        top_k = kwargs.get('top_k')
    else:
        top_k = None
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    for index, data in test.iterrows():
        if top_k == None:
            predict_val = predict(data['image_1'], data['image_2'], model=model)  # predicted sentence
        else:
            predict_val = predict(data['image_1'], data['image_2'], model=model, top_k=top_k)
        true = data.impression
        _ = get_bleu(true, predict_val)
        bleu1.append(_[0])
        bleu2.append(_[1])
        bleu3.append(_[2])
        bleu4.append(_[3])
    return np.array(bleu1).mean(), np.array(bleu2).mean(), np.array(bleu3).mean(), np.array(bleu4).mean()


def greedy_search_predict(image1, image2, model=model1):
    """
    Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
    """
    image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
    image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255
    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST),
                            axis=0)  # introduce batch and resize
    image2 = tf.expand_dims(cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)

    image1 = model.get_layer('image_encoder')(image1)  # output from chexnet
    image2 = model.get_layer('image_encoder')(image2)

    # image1 = model.get_layer('bk_dense')(image1) #op from dense layer
    # image2 = model.get_layer('bk_dense')(image2)

    concat = model.get_layer('concatenate')([image1, image2])
    image_dense = model.get_layer('Image_dense')(concat)
    # concat = model.get_layer('batch_normalization')(concat)
    # image_dense = model.get_layer('Image_dense')(concat)
    bk_feat = tf.keras.backend.expand_dims(image_dense, axis=1)

    states = [image_dense, image_dense]
    a = []
    pred = []
    for i in range(max_pad):
        if i == 0:  # if first word
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))  # shape: (1,1)
        caption = model.get_layer('embedding')(caption)  # embedding shape = 1*1*300
        caption, state_h, state_c = model.get_layer('lstm')(caption, initial_state=states)  # lstm 1*1*512
        states = [state_h, state_c]

        add = model.get_layer('add')([bk_feat, caption])  # add
        output = model.get_layer('output_dense')(add)  # 1*1*vocab_size (here batch_size=1)

        # prediction
        max_prob = tf.argmax(output, axis=-1)  # tf.Tensor of shape = (1,1)
        caption = np.array(max_prob)  # will be sent to embedding for next iteration
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break;
        else:
            a.append(tf.squeeze(max_prob).numpy())
    return tokenizer.sequences_to_texts([a])[0]  # here output would be 1,1 so subscripting to open the array


# k = -1
# image1, image2 = test['image_1'].iloc[k], test['image_2'].iloc[k]
# print(greedy_search_predict(image1, image2, model=model))
#
# _ = mean_bleu(test, greedy_search_predict)
#
# k = list(_)
# index = 'greedy search'
# result = pd.DataFrame([k], columns=["bleu1", "bleu2", "bleu3", "bleu4"], index=[index])
# print(result)


def encoder_op(image1, image2, model=model1):
    """
  Given image1 and image2 filepath, outputs
  their backbone features which will be input
  to the decoder
  """
    image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
    image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255

    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST),
                            axis=0)  # introduce batch and resize
    image2 = tf.expand_dims(cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)

    image1 = model.get_layer('image_encoder')(image1)  # output from chexnet
    image2 = model.get_layer('image_encoder')(image2)

    concat = model.get_layer('concatenate')([image1, image2])
    image_dense = model.get_layer('Image_dense')(concat)
    bk_feat = tf.keras.backend.expand_dims(image_dense, axis=1)
    states = [image_dense, image_dense]
    return bk_feat, states


def beam_search_predict(image1, image2, top_k=3, max_pad=max_pad, model=model1):
    """
    Given image1, image2 get the top
    beam search predicted sentence
    """
    k = top_k
    cls_token = tokenizer.texts_to_sequences(['<cls>'])[0]  # [3]
    bk_feat, states = encoder_op(image1, image2)
    seq_score = [[cls_token, 0, states]]  # [[[3], 0]]
    finished_seq_score = []
    for i in range(max_pad):  # traverse through all lengths
        all_candidates = []  # stores all the top k seq along with their scores
        new_seq_score = []  # stores the seq_score which does not have <end> in them
        for s in seq_score:  # traverse for all top k sequences
            text_input = s[0][-1]  # getting the last predicted output
            # print(s)
            states = s[2]
            caption = model.get_layer('embedding')(
                np.array([[text_input]]))  # ip must be in shape (batch_size,seq length,dim)
            caption, state_h, state_c = model.get_layer('lstm')(caption, initial_state=states)
            states = [state_h, state_c]
            add = model.get_layer('add')([bk_feat, caption])
            output = model.get_layer('output_dense')(add)[0][0]  # (vocab_size,)
            top_words = tf.argsort(output, direction='DESCENDING')[:k]  # get the top k words

            seq, score, _ = s
            for t in top_words.numpy():
                # here we will update score with log of probabilities and subtracting(log of prob will be in negative)
                # here since its -(log), lower the score higher the prob
                candidates = [seq + [t], score - np.log(output[t].numpy()), states]  # updating the score and seq
                all_candidates.append(candidates)
            seq_score = sorted(all_candidates, key=lambda l: l[1])[
                        :k]  # getting the top 3 sentences with high prob ie low score
            # checks for  <end> in each seq obtained
            count = 0
            end_token = tokenizer.word_index['<end>']
            for seq, score, state in seq_score:
                # print('seq,score',seq,score)
                if seq[-1] == end_token:  # if last word of the seq is <end>
                    finished_seq_score.append([seq, score])
                    count += 1
                else:
                    new_seq_score.append([seq, score, state])
            k -= count  # substracting the no. of finished sentences from beam length
            seq_score = new_seq_score

            if seq_score == []:  # if null array
                break;
            else:
                continue;

    seq_score = finished_seq_score[-1]
    sentence = seq_score[0][1:-1]  # here <cls> and <end> is here so not considering that
    score = seq_score[1]

    return tokenizer.sequences_to_texts([sentence])[0]


k = -1
image1, image2 = test['image_1'].iloc[k], test['image_2'].iloc[k]
print(beam_search_predict(image1, image2, top_k=3))

test['bleu_1_gs'] = np.zeros(test.shape[0])  # greedy search
test['bleu_1_bm'] = np.zeros(test.shape[0])  # beam search
test['prediction_gs'] = np.zeros(test.shape[0])  # greedy search
test['prediction_bm'] = np.zeros(test.shape[0])  # beam search
for index, rows in tqdm(test.iterrows(), total=test.shape[0]):
    # greedy search
    predicted_text = greedy_search_predict(rows.image_1, rows.image_2, model1)
    test.loc[index, 'prediction_gs'] = predicted_text
    reference = [rows['impression'].split()]
    test.loc[index, 'bleu_1_gs'] = sentence_bleu(reference, predicted_text.split(), weights=(1, 0, 0, 0))

    # beam search
    predicted_text = beam_search_predict(rows.image_1, rows.image_2, top_k=3, model=model1)
    test.loc[index, 'prediction_bm'] = predicted_text
    test.loc[index, 'bleu_1_bm'] = sentence_bleu(reference, predicted_text.split(), weights=(1, 0, 0, 0))

print(test['prediction_gs'].value_counts() * 100 / test.shape[0])  # greedy search
print(test['prediction_bm'].value_counts() * 100 / test.shape[0])  # beam search


def final_caption_pred(image1, image2, method="beam", top_k=3, model=model1):
    """
    Given image1. image2 paths, the model, top_k and the method of prediction returns the predicted caption
    method: "greedy" or "g" for greedy search, "beam" or "b" for beam search
    """
    if method in ['greedy', 'g']:
        pred_caption = greedy_search_predict(image1, image2, model)
    elif method in ['beam', 'b']:
        pred_caption = beam_search_predict(image1, image2, top_k=top_k, model=model)
    else:
        print("Enter 'b' or 'beam' for beam search and 'g' or 'greedy' for greedy search")

    return pred_caption


def inference(image1, image2, true_caption, model=model1, top_k=[3], image_size=(10, 20)):
    """
  given 2 images (their paths), the true caption, the model and the range of top_k
  prints the two images, true caption along with greedy search prediction and beam search prediction of top_k range
  """
    image1_array = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    image2_array = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    if type(top_k) == int:
        top_k = [top_k]  # changing it to list if top_k given is of int type
    greedy_caption = final_caption_pred(image1, image2, method='g', model=model)  # getting the greedy search prediction

    # printing the 2 images
    plt.figure(figsize=image_size)
    plt.subplot(121)
    plt.imshow(image1_array)
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(image2_array)
    plt.axis("off")
    plt.show()

    print("\nTrue caption: '%s'" % (true_caption))
    print("Predicted caption(greedy search): '%s'" % (greedy_caption))
    # beam search of top_k
    if top_k != None:
        for i in top_k:
            beam_caption = final_caption_pred(image1, image2, method='b', model=model, top_k=i)
            print("Predicted caption(beam search = %i): '%s'" % (i, beam_caption))


i = test[test['bleu_1_gs'] > 0.0].sample(5).index
for k in i:
    image1, image2 = test['image_1'][k], test['image_2'][k]
    true_caption = test['impression'][k]
    inference(image1, image2, true_caption)

print('ok')
