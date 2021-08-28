from base_model import BaseLine
from DataLoader import tokenizing_analysis
from DataLoader import Dataloader, Dataset
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, \
    Embedding, LSTM, Dot, Reshape, Concatenate, BatchNormalization, \
    GlobalMaxPooling2D, Dropout, Add, MaxPooling2D, GRU, AveragePooling2D
from config import config as args
import os


# chexnet weights ; https://drive.google.com/file/d/19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b/view
# 和base line的网络结构不一样
def create_chexnet(chexnet_weights=args.chexnet_weights, input_size=args.input_size):
    """
  chexnet_weights: weights value in .h5 format of chexnet
  creates a chexnet model with preloaded weights present in chexnet_weights file
  """
    # importing densenet the last layer will be a relu activation layer
    model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_size + (3,))

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
    return chexnet


# 和 baseline 的网络结构不一样
class Image_encoder(tf.keras.layers.Layer):
    """
    This layer will output image backbone features after passing it through chexnet
    here chexnet will be not be trainable
    """

    def __init__(self, name="image_encoder_block"):
        super().__init__()
        self.chexnet = create_chexnet(args.chexnet_weights)
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()
        # for i in range(10): #the last 10 layers of chexnet will be trained
        #   self.chexnet.layers[-i].trainable = True

    def call(self, data):
        op = self.chexnet(data)  # op shape: (None,7,7,1024)
        print(op.shape)
        op = self.avgpool(op)  # op shape (None,3,3,1024)
        op = tf.reshape(op, shape=(-1, op.shape[1] * op.shape[2], op.shape[3]))  # op shape: (None,9,1024)
        return op


def encoder(image1, image2, dense_dim=args.dense_dim, dropout_rate=args.dropout_rate):
    """
  Takes image1,image2
  gets the final encoded vector of these
  """
    # image1
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)  # shape: (None,9,1024)
    print('bkfeat1:', bkfeat1.shape)
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

    def __init__(self, dense_dim=args.dense_dim):  # desne_dim = 512
        super().__init__()
        # Intialize variables needed for Concat score function here
        self.W1 = Dense(units=dense_dim)  # weight matrix of shape enc_units*dense_dim
        self.W2 = Dense(units=dense_dim)  # weight matrix of shape dec_units*dense_dim
        self.V = Dense(units=1)  # weight matrix of shape dense_dim*1
        # op (None,98,1)

    # here the encoded output will be the concatted image bk features shape: (None,98,dense_dim)
    def call(self, encoder_output, decoder_h):
        decoder_h = tf.expand_dims(decoder_h, axis=1)  # shape: (None,1,dense_dim)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)  # ouput_shape: batch_size*98*dense_dim
        tanh_output = tf.nn.tanh(tanh_input)
        # shape= batch_size*98*1 getting attention alphas
        attention_weights = tf.nn.softmax(self.V(tanh_output), axis=1)  # 得到attention权重
        # op_shape: batch_size*98*dense_dim  multiply all aplhas with corresponding context vector
        op = attention_weights * encoder_output  # 权重和输出相乘
        # summing all context vector over the time period ie input length, output_shape: batch_size*dense_dim
        context_vector = tf.reduce_sum(op, axis=1)  # 对结果压缩求和，所有元素加起来得到一个
        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
    """
    decodes a single token
    """

    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, embedding_matrix,
                 name="onestepdecoder"):
        # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size + 1,
                                   output_dim=embedding_dim,
                                   input_length=max_pad,
                                   weights=[embedding_matrix],
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
        print('embedding_op.shape:', embedding_op.shape)

        context_vector, attention_weights = self.attention(encoder_output,
                                                           decoder_h)  # passing hidden state h of decoder and encoder output
        print('context_vector:', context_vector.shape)
        print('attention_weights:', attention_weights.shape)
        # context_vector shape: batch_size*dense_dim we need to add time dimension
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        print('context_vector_time_axis:', context_vector_time_axis.shape)
        print('embedding_op:', embedding_op.shape)
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

    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size, embedding_matrix,
                 score_fun='general'):
        super().__init__()
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim
        self.onestepdecoder = One_Step_Decoder(vocab_size=vocab_size,
                                               embedding_dim=embedding_dim,
                                               max_pad=max_pad,
                                               dense_dim=dense_dim,
                                               embedding_matrix=embedding_matrix)

    @tf.function
    def call(self, encoder_output,
             caption):  # ,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(
            encoder_output[:, 0])  # decoder_h, decoder_c
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        print('caption.shape', caption.shape, " encoder_output:", encoder_output.shape, " decoder_h:", decoder_h.shape)
        print('max_pad:', self.max_pad)
        for timestep in range(self.max_pad):  # iterating through all timesteps ie through max_pad
            print('input_caption:', caption[:, timestep:timestep + 1].shape)
            output, decoder_h, attention_weights = self.onestepdecoder(caption[:, timestep:timestep + 1],
                                                                       encoder_output, decoder_h)
            # print(output_array, "output")
            output_array = output_array.write(timestep, output)  # timestep*batch_size*vocab_size

        self.output_array = tf.transpose(output_array.stack(), [1, 0,
                                                                2])  # .stack :Return the values in the TensorArray as a stacked Tensor.)
        # shape output_array: (batch_size,max_pad,vocab_size)
        return self.output_array


class Attention_Model(BaseLine):
    def __init__(self, print_model=True, draw_model=False):  # supper from baseline
        BaseLine.__init__(self, print_model, draw_model)
        self.embedding_matrix = self.word_embedding()
        self.output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        self.decoder = decoder(max_pad=self.max_pad,
                               embedding_dim=self.embedding_dim,
                               dense_dim=self.dense_dim,
                               batch_size=self.batch_size,
                               vocab_size=self.vocab_size,
                               embedding_matrix=self.embedding_matrix)

    def model(self):
        image1 = Input(shape=(self.input_size + (3,)), name="image1")  # shape = 224,224,3
        image2 = Input(shape=(self.input_size + (3,)), name="image2")
        caption = Input(shape=(self.max_pad,))

        encoder_output = encoder(image1, image2)  # shape: (None,28,512)
        print(encoder_output.shape, "====`")
        self.decoder(encoder_output, caption)
        output = self.decoder(encoder_output, caption)
        model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
        print(model.summary())
        # self.train(model)
        self.predict(model)
        # return model




Attention_Model().model()
# Attention_Model().train(model)

