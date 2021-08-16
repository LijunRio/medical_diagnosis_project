import cv2
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenizing_analysis(train, test, visualising=False):
    """
    tokenization 对句子进行分词，并对每个sequence进行分析
    :param train: read train from pkl file
    :param test:  read test from pkl file
    :return: tokenizer, max_pad
    """
    tokenizer = Tokenizer(filters='', oov_token='<unk>')  # setting filters to none
    tokenizer.fit_on_texts(train.impression_final.values)  # '<CLS> ' + df.impression + ' <END>'

    # train.impression : list[string]
    # train_captions: list[list[int]]
    train_captions = tokenizer.texts_to_sequences(train.impression_final)
    test_captions = tokenizer.texts_to_sequences(test.impression_final)
    vocab_size = len(tokenizer.word_index)  # tokenization之后的sequence长度，单词长度
    caption_len = np.array([len(i) for i in train_captions])
    start_index = tokenizer.word_index['<cls>']  # tokened value of <cls>
    end_index = tokenizer.word_index['<end>']  # tokened value of <end>

    # visualising impression length and other details
    # 可视化句子长度
    if visualising:
        ax = sns.displot(caption_len, height=6)
        ax.set_titles('Value Counts vs Caption Length')
        ax.set_xlabels('Impresion length')
        plt.show()

        # 打印top5句子的长度
        print('\nValue Counts for caption length top 5 values\n')
        print('Length|Counts')
        print(pd.Series(caption_len).value_counts()[:5])
        print('\nThe max and min value of "caption length" was found to be %i and %i respectively' % (
            max(caption_len), min(caption_len)))
        print('The 80 percentile value of caption_len which is %i will be taken as '
              'the maximum padded value for each impression' % (np.percentile(caption_len, 80)))
    max_pad = int(np.percentile(caption_len, 80))  # get the max_pad size
    # del train_captions, test_captions  # we will create tokenizing  and padding in-built in dataloader
    return tokenizer, max_pad, test_captions, vocab_size, start_index, end_index


# 输入数据的pipline
class Dataset():
    # here we will get the images converted to vector form and the corresponding captions
    def __init__(self, df, input_size, tokenizer, max_pad, augmentation=True):
        """
        df  = dataframe containing image_1,image_2 and impression
        """
        self.image1 = df.image_1
        self.image2 = df.image_2
        self.caption = df.impression_ip  # inp '<CLS> ' + df.impression
        self.caption1 = df.impression_op  # output   df.impression + ' <END>'
        self.input_size = input_size  # tuple ex: (512,512)
        self.tokenizer = tokenizer  # get from tokenizing_analysis()
        self.max_pad = max_pad  # get from tokenizing_analysis()
        self.augmentation = augmentation  # default is True

        # image augmentation
        # https://imgaug.readthedocs.io/en/latest/source/overview/flip.html?highlight=Fliplr
        self.aug1 = iaa.Fliplr(1)  # flip images horizontally
        self.aug2 = iaa.Flipud(1)  # flip images vertically

    def __getitem__(self, i):
        # gets the datapoint at i th index, we will extract the feature vectors of images after resizing the image
        # and apply augmentation
        image1 = cv2.imread(self.image1[i], cv2.IMREAD_UNCHANGED) / 255  # 保留原图和原有颜色
        image2 = cv2.imread(self.image2[i], cv2.IMREAD_UNCHANGED) / 255  # here there are 3 channels
        image1 = cv2.resize(image1, self.input_size, interpolation=cv2.INTER_NEAREST)  # 最邻近插值进行resize
        image2 = cv2.resize(image2, self.input_size, interpolation=cv2.INTER_NEAREST)
        if image1.any() == None:
            print("%i , %s image sent null value" % (i, self.image1[i]))
        if image2.any() == None:
            print("%i , %s image sent null value" % (i, self.image2[i]))

        # tokenizing and padding
        # caption 是impression部分加上<cls>开始标记
        caption = self.tokenizer.texts_to_sequences(
            self.caption[i:i + 1])  # the input should be an array for tokenizer ie [self.caption[i]]

        caption = pad_sequences(caption, maxlen=self.max_pad, padding='post')  # opshape:(input_length,)
        caption = tf.squeeze(caption, axis=0)  # opshape = (input_length,) removing unwanted axis if present

        # caption1为输出，impression加上<END>结束标志为
        caption1 = self.tokenizer.texts_to_sequences(
            self.caption1[i:i + 1])  # the input should be an array for tokenizer ie [self.caption[i]]

        caption1 = pad_sequences(caption1, maxlen=self.max_pad, padding='post')  # opshape: (input_length,)
        caption1 = tf.squeeze(caption1, axis=0)  # opshape = (input_length,) removing unwanted axis if present

        if self.augmentation:  # we will not apply augmentation that crops the image
            a = np.random.uniform()
            if a < 0.333:
                image1 = self.aug1.augment_image(image1)
                image2 = self.aug1.augment_image(image2)
            elif a < 0.667:
                image1 = self.aug2.augment_image(image1)
                image2 = self.aug2.augment_image(image2)
            else:  # applying no augmentation
                pass;

        return image1, image2, caption, caption1

    def __len__(self):
        return len(self.image1)


# Dataloader继承自tf.keras.utils.Sequence
class Dataloader(tf.keras.utils.Sequence):  # for batching
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batchsize默认1
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset))

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        indexes = [self.indexes[j] for j in range(start, stop)]  # getting the shuffled index values
        data = [self.dataset[j] for j in
                indexes]  # taken from Data class (calls __getitem__ of Data) here the shape is batch_size*3,
        # (image_1,image_2,caption)

        # here the shape will become batch_size*input_size(of image)*3, batch_size*input_size(of image)*3 ,
        # batch_size*1*max_pad
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple([[batch[0], batch[1], batch[2]],
                      batch[3]])  # here [image1,image2, caption(without <END>)],caption(without <CLS>) (op)

    def __len__(self):  # returns total number of batches in an epoch
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):  # it runs at the end of epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)  # in-place shuffling takes place


if __name__ == '__main__':
    #
    #     # Set Hyper parameters
    batch_size = 100
    #     embedding_dim = 300
    #     dense_dim = 512
    #     lstm_units = dense_dim
    #     dropout_rate = 0.2
    #
    # load data
    folder_name = '../pickle_files'
    file_name = 'train.pkl'
    train = pd.read_pickle(os.path.join(folder_name, file_name))
    file_name = 'test.pkl'
    test = pd.read_pickle(os.path.join(folder_name, file_name))

    # DataLoader Part
    input_size = (224, 224)
    tokenizer, max_pad, *_ = tokenizing_analysis(train=train, test=test, visualising=True)
    print("max_pad:", max_pad)
    train_dataloader = Dataset(train, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
    train_dataloader = Dataloader(train_dataloader, batch_size=batch_size)

    test_dataloader = Dataset(test, input_size=input_size, tokenizer=tokenizer, max_pad=max_pad)
    test_dataloader = Dataloader(test_dataloader, batch_size=batch_size)
