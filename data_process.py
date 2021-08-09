import os
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import cv2
from wordcloud import WordCloud, STOPWORDS  # for plotting wordcloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns


def find_association(reports_folder):
    """
    找到报告中链接的X光图片关系，并进行分析
    :return:
    """
    no_images = []  # stores the no. of images
    for file in os.listdir(reports_folder):
        report_file = os.path.join(reports_folder, file)
        with open(report_file, 'r') as f:  # reading the xml data
            data = f.read()  # read the xml file
        regex = r"parentImage id.*"  # getting all the image names
        k = re.findall(regex, data)  # find the parentImage id in xml files
        temp = len(k)  # get the parentImage nums
        no_images.append(temp)
    no_images = np.array(no_images)  # change to matrix
    print('no_images.shape:%i' % no_images.shape)
    print("The max no. of images found associated with a report: %i" % (no_images.max()))
    print("The min no. of images found associated with a report: %i" % (no_images.min()))

    print("Image Value_counts\n")
    print(pd.Series(no_images).value_counts())


def decontracted(phrase):  # https://stackoverflow.com/a/47091490
    """
    去除缩写
    performs text decontraction of words like won't to will not
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_info(xml_data, info):  # https://regex101.com/
    """
    提取报告中的信息
    extracts the information data from the xml file and does text preprocessing on them
    here info can be 1 value in this list ["COMPARISON","INDICATION","FINDINGS","IMPRESSION"]
    """
    regex = r"\"" + info + r"\".*"
    k = re.findall(regex, xml_data)[0]  # finding info part of the report

    regex = r"\>.*\<"
    k = re.findall(regex, k)[0]  # removing info string and /AbstractText>'

    regex = r"\d."
    k = re.sub(regex, "", k)  # removing all values like "1." and "2." etc

    regex = r"X+"
    k = re.sub(regex, "", k)  # removing words like XXXX

    regex = r" \."
    k = re.sub(regex, "", k)  # removing singular fullstop ie " ."

    regex = r"[^.a-zA-Z]"
    k = re.sub(regex, " ", k)  # removing all special characters except for full stop

    regex = r"\."
    k = re.sub(regex, " .", k)  # adding space before fullstop
    k = decontracted(k)  # perform decontraction
    k = k.strip().lower()  # strips the begining and end of the string of spaces and converts all into lowercase
    k = " ".join(k.split())  # removes unwanted spaces
    if k == "":  # if the resulting sentence is an empty string return null value
        k = np.nan
    return k


def get_final(data):
    """
    given an xml data returns "COMPARISON","INDICATION","FINDINGS","IMPRESSION" part of the data
    """
    try:  # assigning null values to the ones that don't have the concerned info
        comparison = get_info(data, "COMPARISON")
    except:
        comparison = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        indication = get_info(data, "INDICATION")
    except:
        indication = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        finding = get_info(data, "FINDINGS")
    except:
        finding = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        impression = get_info(data, "IMPRESSION")
    except:
        impression = np.nan;

    return comparison, indication, finding, impression


def get_df():
    """
    Given an xml data, it will extract the two image names and corresponding info text and returns a dataframe
    """
    im1 = []  # there are 2 images associated with a report
    im2 = []
    # stores info
    comparisons = []
    indications = []
    findings = []
    impressions = []
    report = []  # stores xml file name
    for file in tqdm(os.listdir(reports_folder)):
        report_file = os.path.join(reports_folder, file)
        with open(report_file, 'r') as f:  # reading the xml data
            data = f.read()

        regex = r"parentImage id.*"  # getting all the image names
        k = re.findall(regex, data)

        if len(k) == 2:
            regex = r"\".*\""  # getting the name
            image1 = re.findall(regex, k[0])[0]
            image2 = re.findall(regex, k[1])[0]

            image1 = re.sub(r"\"", "", image1)
            image2 = re.sub(r"\"", "", image2)

            image1 = image1.strip() + ".png"
            image2 = image2.strip() + ".png"
            im1.append(image1)
            im2.append(image2)

            comparison, indication, finding, impression = get_final(data)
            comparisons.append(comparison)
            indications.append(indication)
            findings.append(finding)
            impressions.append(impression)
            report.append(file)  # xml file name

        elif len(k) < 2:  # 如果一份报告链接的图片少于两张
            regex = r"\".*\""  # getting the name
            try:  # if the exception is raised means no image file name was found
                image1 = re.findall(regex, k[0])[0]
                image1 = re.sub(r"\"", "", image1)  # removing "
                image2 = np.nan

                image1 = image1.strip() + ".png"
            except:
                image1 = np.nan
                image2 = np.nan

            im1.append(image1)
            im2.append(image2)
            comparison, indication, finding, impression = get_final(data)
            comparisons.append(comparison)
            indications.append(indication)
            findings.append(finding)
            impressions.append(impression)
            report.append(file)  # xml file name

        # if there are more than 2 images concerned with report
        # creat new datapoint with new image and same info
        else:
            comparison, indication, finding, impression = get_final(data)

            for i in range(len(k) - 1):
                regex = r"\".*\""  # getting the name
                image1 = re.findall(regex, k[i])[0]  # re.findall returns a list
                image2 = re.findall(regex, k[i + 1])[0]

                image1 = re.sub(r"\"", "", image1)  # removing "
                image2 = re.sub(r"\"", "", image2)  # removing "

                image1 = image1.strip() + ".png"
                image2 = image2.strip() + ".png"

                im1.append(image1)
                im2.append(image2)
                comparisons.append(comparison)
                indications.append(indication)
                findings.append(finding)
                impressions.append(impression)
                report.append(file)  # xml file name

    df = pd.DataFrame(
        {"image_1": im1, "image_2": im2, "comparison": comparisons, "indication": indications, "findings": findings,
         "impression": impressions, "xml file name": report})
    return df


def process_missiing_value(df, image_folder):
    """
    processing the missing value in the raw dataframe
    :param df: read the df dataframe
    :return: save the final_data into pkl file
    """
    # 查看缺失值所占的比例
    print('df.shape:', df.shape)
    print("columns\t\t%missing values")
    print('-' * 30)
    print(df.isnull().sum() * 100 / df.shape[0])  # percentage missing values

    # 对于image_1和imporession都小于5%，直接删除
    df.drop(df[(df['impression'].isnull()) | (df['image_1'].isnull())].index, inplace=True)
    df = df.reset_index(drop=True).copy()
    print("%i datapoints were removed.\nFinal no. of datapoints: %i" % (4169 - df.shape[0], df.shape[0]))

    # image_2中缺失的行，我们可以直接将复制一张image1到image2
    # this process cost lots of time
    df.loc[df.image_2.isnull(), 'image_2'] = df[df.image_2.isnull()]['image_1'].values
    im1_size = []
    im2_size = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            im1_size.append(cv2.imread(os.path.join(image_folder, row.get('image_1'))).shape[:2])
            im2_size.append(cv2.imread(os.path.join(image_folder, row.get('image_2'))).shape[:2])
        except :
            print(os.path.join(image_folder, row.get('image_1')))
            print(os.path.join(image_folder, row.get('image_2')))

    df['im1_height'] = [i[0] for i in im1_size]
    df['im1_width'] = [i[1] for i in im1_size]
    df['im2_height'] = [i[0] for i in im2_size]
    df['im2_width'] = [i[1] for i in im2_size]

    # save the final data to pkl file
    df.to_pickle("../data/pickle_files/df_final.pkl")
    print('-' * 30)
    print('final data shape:', df.shape)
    print('after processing missing value')
    print(df.head(2))


def evaluate_height_weight(df):
    print('df.shape:', df.shape)
    print("\n\nValue Counts of image_1 heights:\n")
    print(df.im1_height.value_counts()[:5])
    print("\n", "*" * 50, "\n")
    print("Value Counts of image_2 heights:\n")
    print(df.im2_height.value_counts()[:5])
    """
    we can observe that 430 is the most common height for image_1
    while for image_2, the most common height is 624.
    The next common for both of the images is 512
    """
    print("Value Counts of image_1 widths:\n")
    print(df.im1_width.value_counts()[:5])
    print("\n", "*" * 50, "\n")
    print("Value Counts of image_2 widths:\n")
    print(df.im2_width.value_counts()[:5])


def wordcloud_analysis(df):
    """
    using word cloud to impression the medical reports
    """

    # getting wordclouds
    # https://www.geeksforgeeks.org/generating-word-cloud-python/
    # removing all fullstops and storing the result in a temp variable
    temp = df.loc[:, 'impression'].str.replace(".", "").copy()
    words = ""
    for i in temp.values:
        k = i.split()
        words += " ".join(k) + " "
    word = words.strip()
    wc = WordCloud(width=1024, height=720,
                   background_color='white',
                   stopwords=STOPWORDS,
                   min_font_size=15, ).generate(words)

    del k, words, temp  # 删除变量，但不删除数据
    plt.figure(figsize=(16, 16))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def split_dataset(df, folder_name):
    col = ['image_1', 'image_2', 'impression', 'xml file name']
    df = df[col].copy()
    # path
    df['image_1'] = df['image_1'].apply(
        lambda row: os.path.join(image_folder, row))  # https://stackoverflow.com/a/61880790
    df['image_2'] = df['image_2'].apply(lambda row: os.path.join(image_folder, row))

    df['impression_final'] = '<CLS> ' + df.impression + ' <END>'
    df['impression_ip'] = '<CLS> ' + df.impression
    df['impression_op'] = df.impression + ' <END>'

    df.drop_duplicates(subset=['xml file name'], inplace=True)
    # adding a new column impression counts which tells the total value counts of impression of that datapoint
    k = df['impression'].value_counts()
    df = df.merge(k, left_on='impression', right_index=True)  # join left impression value with right index

    df.columns = ['impression', 'image_1', 'image_2', 'impression_x', 'xml file name', 'impression_final',
                  'impression_ip', 'impression_op', 'impression_counts']  # changin column names
    del df['impression_x']  # deleting impression_x column

    other1 = df[df['impression_counts'] > 5]  # selecting those datapoints which have impression valuecounts >5
    other2 = df[df['impression_counts'] <= 5]  # selecting those datapoints which have impression valuecounts <=5
    train, test = train_test_split(other1, stratify=other1['impression'].values, test_size=0.1, random_state=420)
    test_other2_sample = other2.sample(int(0.2 * other2.shape[0]),
                                       random_state=420)  # getting some datapoints from other2 data for test data
    other2 = other2.drop(test_other2_sample.index, axis=0)
    # here i will be choosing 0.5 as the test size as to create a reasonable size of test data
    test = test.append(test_other2_sample)
    test = test.reset_index(drop=True)

    train = train.append(other2)
    train = train.reset_index(drop=True)
    # train.shape[0], test.shape[0] (3257, 563)此时训练集测试集的大小

    # 上采样、下采样
    df_majority = train[train['impression_counts'] >= 100]  # having value counts >=100
    df_minority = train[train['impression_counts'] <= 5]  # having value counts <=5
    df_other = train[(train['impression_counts'] > 5) & (train['impression_counts'] < 100)]
    # value counts between 5# and 100

    n1 = df_minority.shape[0]
    n2 = df_majority.shape[0]
    n3 = df_other.shape[0]
    # we will upsample them to 30
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=3 * n1,
                                     random_state=420)
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=n2 // 15,
                                       random_state=420)
    df_other_downsampled = resample(df_other,
                                    replace=False,
                                    n_samples=n3 // 10,
                                    random_state=420)

    train = pd.concat([df_majority_downsampled, df_minority_upsampled, df_other_downsampled])
    train = train.reset_index(drop=True)
    del df_minority_upsampled, df_minority, df_majority, df_other, df_other_downsampled
    # train.shape (4487, 8)

    file_name = 'train.pkl'
    train.to_pickle(os.path.join(folder_name, file_name))

    file_name = 'test.pkl'
    test.to_pickle(os.path.join(folder_name, file_name))

    print('split_dataset ok!')


if __name__ == '__main__':
    # load image folder
    image_folder = '../data/image'  # path to folder containing images
    total_images = len(os.listdir(image_folder))
    print('The number of images in data are: %i' % total_images)

    # load reports folder
    reports_folder = "../data/ecgen-radiology"
    total_reports = len(os.listdir(reports_folder))
    print('The number of reports in the data are: %i' % (total_reports))

    # find_association(reports_folder)  # find the association between image and reports
    df = get_df()
    print('No1. get dataframe finsh ! ')
    process_missiing_value(df, image_folder)
    print('No2. processing missing value and save final pikle file!')

    # data impression
    df_final = pd.read_pickle('../data/pickle_files/df_final.pkl')
    # evaluate_height_weight(df)
    # wordcloud_analysis(df)

    folder_name = '../pickle_files'
    split_dataset(df=df_final, folder_name=folder_name)
    print('No3. read final pikle file and split dataset')
