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


# chexnet作为decoder的bacbone
def create_chexnet(chexnet_weights):
    """
    输入chexnet的模型文件，返回chexknet的网络结构
    backbone 不训练
    :param chexnet_weights:
    :return: model layers
    """
    # chexk net就是一个121的全连接网络， 不要最后输出层
    model = tf.keras.applications.DenseNet121(include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)  # 添加一个平均池化
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)  # 添加一个全连接层
    # here activation is sigmoid as seen in research paper

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)  # 将checknet的权重加载进去
    # 这里使用的是倒数第二层，不要最后一个全连接层
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[-2].output)
    return chexnet


# Image Encoder类，继承于tf.keras.layers.Layer
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


class BaseLine(object):
    def __init__(self, print_model=True, draw_model=False):
        self.embedding_dim = args.embedding_dim  # 300
        self.dense_dim = args.dense_dim  # 512
        self.lstm_units = args.lstm_units  # 512
        self.dropout_rate = args.dropout_rate
        self.input_size = args.input_size
        self.lr = args.lr
        self.batch_size = args.batch_size  # 100

        self.train_df = pd.read_pickle(os.path.join(args.data_folder, 'train.pkl'))
        self.test_df = pd.read_pickle(os.path.join(args.data_folder, 'test.pkl'))

        self.tokenizer, self.max_pad, self.test_captions, self.vocab_size, self.start_index, self.end_index = \
            tokenizing_analysis(train=self.train_df, test=self.test_df)
        self.Image_encoder = Image_encoder()

        self.print_model = print_model
        self.draw_model = draw_model
        print('init ok')

    def word_embedding(self):
        """
        使用glove初始化嵌入曾矩阵
        :return: embedding_matrix
        """
        embedding_matrix = np.zeros((self.vocab_size + 1, self.embedding_dim))

        # 使用glove来初始化embbedding层参数
        glove = {}  # glove用于将词向量化
        with open(args.glove_path, encoding='utf-8') as f:
            for line in f:
                # it is stored as string like this "'the': '.418 0.24968 -0.41242 0.1217 0.34527
                # -0.044457 -0.4"
                word = line.split()
                glove[word[0]] = np.asarray(word[1:], dtype='float32')

        # 将glove中的值替换到embedding中
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = glove.get(word)
            if embedding_vector is not None:  # if the word is found in glove vectors
                embedding_matrix[i] = embedding_vector[:self.embedding_dim]

        return embedding_matrix

    def model(self):
        image1 = Input(shape=(self.input_size + (3,)), name="image1")  # shape = 224,224,3
        image2 = Input(shape=(self.input_size + (3,)), name="image2")
        caption = Input(shape=(self.max_pad,), name="caption")  # 第80百分位的长度，28

        """
        图片处理部分
        """
        # 使用chexnet进行图片编码
        bk_feat1 = self.Image_encoder(image1)  # 对image1进行编码
        bk_feat2 = self.Image_encoder(image2)  # 对image2 进行编码

        bk_features_concat = Concatenate(axis=-1)([bk_feat1, bk_feat2])  # (None, 2048)
        image_dense = Dense(self.dense_dim, activation='relu', name='Image_dense', use_bias='False')  # 将2048压缩成512维

        # 将concat到一起的向量再通过dense net
        image_bkbone = image_dense(bk_features_concat)  # 输出（None, 512)
        image_dense_op = tf.keras.backend.expand_dims(image_bkbone, axis=1)  # op_shape: (None,1,512)  # 扩张一维

        """
        文本处理部分
        """
        # 嵌入层 使用glove vector来初始化权重
        embedding_matrix = self.word_embedding()
        embedding = Embedding(input_dim=self.vocab_size + 1,
                              output_dim=self.embedding_dim,
                              input_length=self.max_pad,
                              mask_zero=True,
                              weights=[embedding_matrix],
                              name='embedding'
                              )
        # （None, 28, 300）
        embed_op = embedding(caption)  # op_shape: (None,input_length=28,embedding_dim=300)
        lstm_layer = LSTM(units=self.lstm_units, return_sequences=True, return_state=True)
        # lstm_op (None, 28, 512), lstm_h(None, 512), lstm_c(None, 512)
        lstm_op, lstm_h, lstm_c = lstm_layer(embed_op, initial_state=[image_bkbone, image_bkbone])

        """
        NLP CV融合
        """
        add = Add()([image_dense_op, lstm_op])  # (None, 28+1, 512)
        op_dense = Dense(self.vocab_size + 1,  # 单词长度+1
                         activation='softmax',
                         name='output_dense'
                         )  # op: (None,input_length=29,vocab_size+1=1326)
        output = op_dense(add)
        model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)

        if self.print_model is True:
            print(model.summary())
        if self.draw_model is True:
            # 保存模型结构图
            model_png = args.modelPng_save
            tf.keras.utils.plot_model(model, to_file=model_png, show_shapes=True)

        return model, image1, image2, caption, output

    def train(self):
        model, *_ = self.model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)  # optimizer
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        tb_filename = 'Simple_Encoder_Decoder/'
        tb_file = os.path.join(args.modelSave_path, tb_filename)
        model_filename = 'Simple_Encoder_Decoder2.h5'
        model_save = os.path.join(args.modelSave_path, model_filename)

        my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, verbose=2),
                        tf.keras.callbacks.ModelCheckpoint(filepath=model_save, save_best_only=True,
                                                           save_weights_only=True, verbose=2),
                        tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=tb_file),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                                             min_lr=10 ** -7, verbose=2)]  # from keras documentation

        train_dataloader = Dataset(self.train_df, input_size=self.input_size, tokenizer=self.tokenizer,
                                   max_pad=self.max_pad)
        train_dataloader = Dataloader(train_dataloader, batch_size=self.batch_size)
        test_dataloader = Dataset(self.test_df, input_size=self.input_size, tokenizer=self.tokenizer,
                                  max_pad=self.max_pad)
        test_dataloader = Dataloader(test_dataloader, batch_size=self.batch_size)

        with tf.device("/device:GPU:0"):
            model.fit(train_dataloader,
                      validation_data=test_dataloader,
                      epochs=10,
                      callbacks=my_callbacks
                      )

    def get_bleu(self, reference, prediction):
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
    def mean_bleu(self, test, predict, model, **kwargs):
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
            _ = self.get_bleu(true, predict_val)
            bleu1.append(_[0])
            bleu2.append(_[1])
            bleu3.append(_[2])
            bleu4.append(_[3])
        return np.array(bleu1).mean(), np.array(bleu2).mean(), np.array(bleu3).mean(), np.array(bleu4).mean()

    def encoder_op(self, image1, image2, model):
        """
      Given image1 and image2 filepath, outputs
      their backbone features which will be input
      to the decoder
      """
        image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
        image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255

        image1 = tf.expand_dims(cv2.resize(image1, self.input_size, interpolation=cv2.INTER_NEAREST),
                                axis=0)  # introduce batch and resize
        image2 = tf.expand_dims(cv2.resize(image2, self.input_size, interpolation=cv2.INTER_NEAREST), axis=0)

        image1 = model.get_layer('image_encoder')(image1)  # output from chexnet
        image2 = model.get_layer('image_encoder')(image2)

        concat = model.get_layer('concatenate')([image1, image2])
        image_dense = model.get_layer('Image_dense')(concat)
        bk_feat = tf.keras.backend.expand_dims(image_dense, axis=1)
        states = [image_dense, image_dense]
        return bk_feat, states

    def greedy_search_predict(self, image1, image2, model):
        """
        Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
        """
        bk_feat, states = self.encoder_op(image1, image2, model)
        a = []
        pred = []
        for i in range(self.max_pad):
            if i == 0:  # if first word
                caption = np.array(self.tokenizer.texts_to_sequences(['<cls>']))  # shape: (1,1)
            caption = model.get_layer('embedding')(caption)  # embedding shape = 1*1*300
            caption, state_h, state_c = model.get_layer('lstm')(caption, initial_state=states)  # lstm 1*1*512
            states = [state_h, state_c]

            add = model.get_layer('add')([bk_feat, caption])  # add
            output = model.get_layer('output_dense')(add)  # 1*1*vocab_size (here batch_size=1)

            # prediction
            max_prob = tf.argmax(output, axis=-1)  # tf.Tensor of shape = (1,1)
            caption = np.array(max_prob)  # will be sent to embedding for next iteration
            if max_prob == np.squeeze(self.tokenizer.texts_to_sequences(['<end>'])):
                break;
            else:
                a.append(tf.squeeze(max_prob).numpy())
        return self.tokenizer.sequences_to_texts([a])[0]  # here output would be 1,1 so subscripting to open the array

    def beam_search_predict(self, image1, image2, model, top_k=3):
        """
        Given image1, image2 get the top
        beam search predicted sentence
        """
        k = top_k
        max_pad = self.max_pad
        cls_token = self.tokenizer.texts_to_sequences(['<cls>'])[0]  # [3]
        bk_feat, states = self.encoder_op(image1, image2, model)
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
                end_token = self.tokenizer.word_index['<end>']
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
        return self.tokenizer.sequences_to_texts([sentence])[0]

    def final_caption_pred(self, image1, image2, model, method="beam", top_k=3):
        """
        Given image1. image2 paths, the model, top_k and the method of prediction returns the predicted caption
        method: "greedy" or "g" for greedy search, "beam" or "b" for beam search
        """
        if method in ['greedy', 'g']:
            pred_caption = self.greedy_search_predict(image1, image2, model)
        elif method in ['beam', 'b']:
            pred_caption = self.beam_search_predict(image1, image2, top_k=top_k, model=model)
        else:
            print("Enter 'b' or 'beam' for beam search and 'g' or 'greedy' for greedy search")

        return pred_caption

    def predict_inference(self, image1, image2, true_caption, model, top_k=[3], image_size=(10, 20)):
        """
        given 2 images (their paths), the true caption, the model and the range of top_k
        prints the two images, true caption along with greedy search prediction and beam search prediction of top_k range
        """
        image1_array = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
        image2_array = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
        if type(top_k) == int:
            top_k = [top_k]  # changing it to list if top_k given is of int type
        greedy_caption = self.final_caption_pred(image1, image2, method='g', model=model)

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
        if top_k is not None:
            for i in top_k:
                beam_caption = self.final_caption_pred(image1, image2, method='b', model=model, top_k=i)
                print("Predicted caption(beam search = %i): '%s'" % (i, beam_caption))

    def predict(self, model):
        test = self.test_df

        k = -1
        image1, image2 = test['image_1'].iloc[k], test['image_2'].iloc[k]
        print(self.beam_search_predict(image1, image2, model=model, top_k=3))

        test['bleu_1_gs'] = np.zeros(test.shape[0])  # greedy search
        test['bleu_1_bm'] = np.zeros(test.shape[0])  # beam search
        test['prediction_gs'] = np.zeros(test.shape[0])  # greedy search
        test['prediction_bm'] = np.zeros(test.shape[0])  # beam search
        for index, rows in tqdm(test.iterrows(), total=test.shape[0]):
            # greedy search
            predicted_text = self.greedy_search_predict(rows.image_1, rows.image_2, model=model)
            test.loc[index, 'prediction_gs'] = predicted_text
            reference = [rows['impression'].split()]
            test.loc[index, 'bleu_1_gs'] = sentence_bleu(reference, predicted_text.split(), weights=(1, 0, 0, 0))

            # beam search
            predicted_text = self.beam_search_predict(rows.image_1, rows.image_2, model=model, top_k=3)
            test.loc[index, 'prediction_bm'] = predicted_text
            test.loc[index, 'bleu_1_bm'] = sentence_bleu(reference, predicted_text.split(), weights=(1, 0, 0, 0))

        print(test['prediction_gs'].value_counts() * 100 / test.shape[0])  # greedy search
        print(test['prediction_bm'].value_counts() * 100 / test.shape[0])  # beam search


if __name__ == '__main__':
    baseline = BaseLine(print_model=True, draw_model=True)
    baseline.model()
    # baseline.train()
    # _, image1, image2, caption,output = baseline.model()
    # model1 = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
    # model1.load_weights(os.path.join(args.modelSave_path, 'Simple_Encoder_Decoder_0.h5'))
    # print(model1.layers)
    # baseline.predict(model=model1)
