import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, Concatenate, \
    BatchNormalization, Dropout, Add, GRU, AveragePooling2D
from config import config as args
import os
from DataLoader import Dataloader, Dataset
from DataLoader import tokenizing_analysis
from tqdm import tqdm

chexnet_weights = os.path.join(args.chexnet_weights, 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5')


def create_chexnet(chexnet_weights=chexnet_weights, input_size=(224, 224)):
    """
  chexnet_weights: weights value in .h5 format of chexnet
  creates a chexnet model with preloaded weights present in chexnet_weights file
  """
    print('chexnet_weights:', chexnet_weights)
    model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_size + (3,))
    # importing densenet the last layer will be a relu activation layer

    # we need to load the weights so setting the architecture of the model as same as the one of the chexnet
    x = model.output  # output from chexnet
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(
        x)  # here activation is sigmoid as seen in research paper

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[
        -3].output)  # we will be taking the 3rd last layer (here it is layer before global avgpooling)
    # since we are using attention here
    print('create_checknet!')
    return chexnet


class Image_encoder(tf.keras.layers.Layer):
    """
  This layer will output image backbone features after passing it through chexnet
  """

    def __init__(self,
                 name="image_encoder_block"
                 ):
        super().__init__()
        self.chexnet = create_chexnet(input_size=(224, 224))
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()
        # for i in range(10): #the last 10 layers of chexnet will be trained
        #   self.chexnet.layers[-i].trainable = True

    def call(self, data):
        op = self.chexnet(data)  # op shape: (None,7,7,1024)
        op = self.avgpool(op)  # op shape (None,3,3,1024)
        op = tf.reshape(op, shape=(-1, op.shape[1] * op.shape[2], op.shape[3]))  # op shape: (None,9,1024)
        return op


def encoder(image1, image2, dense_dim, dropout_rate):
    """
  Takes image1,image2
  gets the final encoded vector of these
  """
    # image1
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)  # shape: (None,9,1024)
    bk_dense = Dense(dense_dim, name='bkdense', activation='relu')  # shape: (None,9,512)
    bkfeat1 = bk_dense(bkfeat1)

    # image2
    bkfeat2 = im_encoder(image2)  # shape: (None,9,1024)
    bkfeat2 = bk_dense(bkfeat2)  # shape: (None,9,512)

    # combining image1 and image2
    concat = Concatenate(axis=1)([bkfeat1, bkfeat2])  # concatenating through the second axis shape: (None,18,1024)
    bn = BatchNormalization(name="encoder_batch_norm")(concat)
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout


class global_attention(tf.keras.layers.Layer):
    """
  calculate global attention
  """

    def __init__(self, dense_dim):
        super().__init__()
        # Intialize variables needed for Concat score function here
        self.W1 = Dense(units=dense_dim)  # weight matrix of shape enc_units*dense_dim
        self.W2 = Dense(units=dense_dim)  # weight matrix of shape dec_units*dense_dim
        self.V = Dense(units=1)  # weight matrix of shape dense_dim*1
        # op (None,98,1)

    def call(self, encoder_output,
             decoder_h):  # here the encoded output will be the concatted image bk features shape: (None,98,dense_dim)
        decoder_h = tf.expand_dims(decoder_h, axis=1)  # shape: (None,1,dense_dim)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)  # ouput_shape: batch_size*98*dense_dim
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output),
                                          axis=1)  # shape= batch_size*98*1 getting attention alphas
        op = attention_weights * encoder_output  # op_shape: batch_size*98*dense_dim  multiply all aplhas with corresponding context vector
        context_vector = tf.reduce_sum(op,
                                       axis=1)  # summing all context vector over the time period ie input length, output_shape: batch_size*dense_dim

        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
    """
  decodes a single token
  """

    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, name="onestepdecoder"):
        # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size + 1,
                                   output_dim=embedding_dim,
                                   input_length=max_pad,
                                   mask_zero=True,
                                   name='onestepdecoder_embedding'
                                   )
        self.LSTM = GRU(units=self.dense_dim,
                        # return_sequences=True,
                        return_state=True,
                        name='onestepdecoder_LSTM'
                        )
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(dense_dim, name='onestepdecoder_embedding_dense', activation='relu')
        self.final = Dense(vocab_size + 1, activation='softmax')
        self.concat = Concatenate(axis=-1)
        self.add = Add()

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):  # ,decoder_c):
        '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B

      here state_h,state_c are decoder states
    '''
        embedding_op = self.embedding(input_to_decoder)  # output shape = batch_size*1*embedding_shape (only 1 token)

        context_vector, attention_weights = self.attention(encoder_output,
                                                           decoder_h)  # passing hidden state h of decoder and encoder output
        # context_vector shape: batch_size*dense_dim we need to add time dimension
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        # now we will combine attention output context vector with next word input to the lstm here we will be teacher forcing
        concat_input = self.concat([context_vector_time_axis,
                                    embedding_op])  # output dimension = batch_size*input_length(here it is 1)*(dense_dim+embedding_dim)

        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        # output shape = batch*1*dense_dim and decoder_h,decoder_c has shape = batch*dense_dim
        # we need to remove the time axis from this decoder_output

        output = self.final(output)  # shape = batch_size*decoder vocab size
        return output, decoder_h, attention_weights


class decoder(tf.keras.Model):
    """
  Decodes the encoder output and caption
  """

    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, max_pad=max_pad,
                                               dense_dim=dense_dim)
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim

    @tf.function
    def call(self, encoder_output, caption):
        # ,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(
            encoder_output[:, 0])  # decoder_h, decoder_c
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):  # iterating through all timesteps ie through max_pad
            output, decoder_h, attention_weights = self.onestepdecoder(caption[:, timestep:timestep + 1],
                                                                       encoder_output, decoder_h)
            output_array = output_array.write(timestep, output)  # timestep*batch_size*vocab_size

        self.output_array = tf.transpose(output_array.stack(), [1, 0, 2])
        # .stack :Return the values in the TensorArray as a stacked Tensor.)
        # shape output_array: (batch_size,max_pad,vocab_size)
        return self.output_array


def create_model(train_flag=False):
    """
  creates the best model ie the attention model
  and returns the model after loading the weights
  and also the tokenizer
  """
    # hyperparameters
    input_size = args.input_size
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    dense_dim = args.dense_dim
    dropout_rate = args.dropout_rate
    train_df = pd.read_pickle(os.path.join(args.finalPkl_ph, 'train.pkl'))
    test_df = pd.read_pickle(os.path.join(args.finalPkl_ph, 'test.pkl'))

    tokenizer, max_pad, test_captions, vocab_size, start_index, end_index = tokenizing_analysis(train_df, test_df)

    tf.keras.backend.clear_session()
    image1 = Input(shape=(input_size + (3,)))  # shape = 224,224,3
    image2 = Input(
        shape=(input_size + (3,)))  # https://www.w3resource.com/python-exercises/tuple/python-tuple-exercise-5.php
    caption = Input(shape=(max_pad,))

    encoder_output = encoder(image1, image2, dense_dim, dropout_rate)  # shape: (None,28,512)

    output = decoder(max_pad, embedding_dim, dense_dim, batch_size, vocab_size)(encoder_output, caption)
    model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
    if not train_flag:
        model_filename = 'Encoder_Decoder_global_attention.h5'
        model_save = os.path.join(args.modelSave_path, model_filename)
        print('load_model_from:', model_save)
        model.load_weights(model_save)
        return model, tokenizer

    return model, tokenizer, max_pad


def greedy_search_predict(image1, image2, model, tokenizer, input_size=args.input_size):
    """
  Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
  """
    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST),
                            axis=0)  # introduce batch and resize
    image2 = tf.expand_dims(cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image1 = model.get_layer('image_encoder')(image1)
    image2 = model.get_layer('image_encoder')(image2)
    image1 = model.get_layer('bkdense')(image1)
    image2 = model.get_layer('bkdense')(image2)

    concat = model.get_layer('concatenate')([image1, image2])
    enc_op = model.get_layer('encoder_batch_norm')(concat)
    enc_op = model.get_layer('encoder_dropout')(enc_op)  # this is the output from encoder

    decoder_h, decoder_c = tf.zeros_like(enc_op[:, 0]), tf.zeros_like(enc_op[:, 0])
    a = []
    pred = []
    # max_pad = args.max_pad
    max_pad = 28  # 暂时写死
    for i in range(max_pad):
        if i == 0:  # if first word
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))  # shape: (1,1)
        output, decoder_h, attention_weights = model.get_layer('decoder').onestepdecoder(caption, enc_op,
                                                                                         decoder_h)  # ,decoder_c) decoder_c,

        # prediction
        max_prob = tf.argmax(output, axis=-1)  # tf.Tensor of shape = (1,1)
        caption = np.array([max_prob])  # will be sent to onstepdecoder for next iteration
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break;
        else:
            a.append(tf.squeeze(max_prob).numpy())
    return tokenizer.sequences_to_texts([a])[0]  # here output would be 1,1 so subscripting to open the array


def get_bleu(reference, prediction):
    """
  Given a reference and prediction string, outputs the 1-gram,2-gram,3-gram and 4-gram bleu scores
  """
    reference = [reference.split()]  # should be in an array (cos of multiple references can be there here only 1)
    prediction = prediction.split()
    bleu1 = sentence_bleu(reference, prediction, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def train():
    model, tokenizer, max_pad = create_model(train_flag=True)
    train_df = pd.read_pickle(os.path.join(args.finalPkl_ph, 'train.pkl'))
    test_df = pd.read_pickle(os.path.join(args.finalPkl_ph, 'test.pkl'))
    print('model--get!')
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)  # optimizer
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # model_save =  os.path.join(args.modelSave_path, 'Encoder_Decoder_global_attention.h5')
    model_save = args.modelSave_path
    print('model save path:', model_save)
    # verbose=2每个epoch输出一行记录
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, verbose=2),  # patience 训练将停止后没有改进的epoch数
                    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.modelSave_path,
                                                                             'Encoder_Decoder_global_attention.h5'),
                                                       save_best_only=True,
                                                       save_weights_only=True, verbose=2),
                    tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=os.path.join(args.modelSave_path,
                                                                                          'Simple_Encoder_Decoder/')),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                                         min_lr=10 ** -7, verbose=2)]  # from keras documentation
    train_dataloader = Dataset(train_df, input_size=args.input_size, tokenizer=tokenizer,
                               max_pad=max_pad)
    train_dataloader = Dataloader(train_dataloader, batch_size=args.batch_size)
    test_dataloader = Dataset(test_df, input_size=args.input_size, tokenizer=tokenizer,
                              max_pad=max_pad)
    test_dataloader = Dataloader(test_dataloader, batch_size=args.batch_size)

    with tf.device("/device:GPU:0"):
        model.fit(train_dataloader,
                  validation_data=test_dataloader,
                  epochs=10,
                  callbacks=my_callbacks
                  )


def predict1(image1, image2=None, model_tokenizer=None):
    """given image1 and image 2 filepaths returns the predicted caption,
  the model_tokenizer will contain stored model_weights and tokenizer 
  """
    if image2 is None:  # if only 1 image file is given
        image2 = image1

    try:
        image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
        image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255
    except:
        return print("Must be an image")

    if model_tokenizer == None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)

    return predicted_caption


def predict2(true_caption, image1, image2=None, model_tokenizer=None):
    """given image1 and image 2 filepaths and the true_caption
   returns the mean of cumulative ngram bleu scores where n=1,2,3,4,
  the model_tokenizer will contain stored model_weights and tokenizer 
  """
    if image2 == None:  # if only 1 image file is given
        image2 = image1

    try:
        image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
        image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255
    except:
        return print("Must be an image")

    if model_tokenizer == None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)

    blue_score = get_bleu(true_caption, predicted_caption)
    blue_score = list(blue_score)
    # final = blue_score.append(predicted_caption)
    return pd.DataFrame([blue_score], columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])


def function1(image1, image2, model_tokenizer=None):
    """
  here image1 and image2 will be a list of image
  filepaths and outputs the resulting captions as a list
  """
    if model_tokenizer is None:
        model_tokenizer = list(create_model())
    predicted_caption = []
    for i1, i2 in zip(image1, image2):
        caption = predict1(i1, i2, model_tokenizer)
        print(caption)
        predicted_caption.append(caption)

    return predicted_caption


def function2(true_caption, image1, image2):
    """
  here true_caption,image1 and image2 will be a list of true_captions and image
  filepaths and outputs the resulting bleu_scores
  as a dataframe
  """
    model_tokenizer = list(create_model())
    predicted = pd.DataFrame(columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])
    for c, i1, i2 in zip(true_caption, image1, image2):
        caption = predict2(c, i1, i2, model_tokenizer)
        predicted = predicted.append(caption, ignore_index=True)

    return predicted


def predict_with_detail(true_caption, image1, image2=None, model_tokenizer=None):
    if image2 is None:  # if only 1 image file is given
        image2 = image1

    try:
        image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255
        image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255
    except:
        return print("Must be an image")

    if model_tokenizer is None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)

    blue_score = get_bleu(true_caption, predicted_caption)
    blue_score = list(blue_score)

    return blue_score, predicted_caption


def get_detail_result(image1, image2, true_caption, model_tokenizer=None):
    if model_tokenizer is None:
        model_tokenizer = list(create_model())
    result = pd.DataFrame(columns=['image1', 'image2', 'GT', 'Predict',
                                   'bleu1', 'bleu2', 'bleu3', 'bleu4'])

    for c, i1, i2 in tqdm(zip(true_caption, image1, image2)):
        im1 = i1.split('/')[-1]
        im2 = i2.split('/')[-1]
        print(c)
        blue_score, predicted_caption = predict_with_detail(c, i1, i2, model_tokenizer)
        final = [im1, im2, c, predicted_caption]
        final.extend(blue_score)
        result = result.append(pd.DataFrame([final], columns=['image1', 'image2', 'GT', 'Predict',
                                                              'bleu1', 'bleu2', 'bleu3', 'bleu4']), ignore_index=True)
    return result

# train()
# create_chexnet()
