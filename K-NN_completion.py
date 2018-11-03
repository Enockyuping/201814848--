'''
VSM-K-NN 完全版
80%训练集，20%测试集
'''
import os
import json
import numpy
import chardet  #用于操作文档编码格式
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from tkinter import _flatten #用于将二维列表转换成一维列表
import collections

def write_file(path, data):
    with open(path, 'w', errors='ignore') as file:
        file.write(data)
def write_dict(path, dict):
    json_dict = json.dumps(dict)
    write_file(path, json_dict)
def read_dict(path):
    with open(path, 'r') as f:
        dict = eval(f.read())
    return dict

def open_file(path):
    '''
    读取文件内容,返回存储文档的字符串
    :parameter:path（文档路径）
    :return:data(文档字符串)
    '''
    with open(path, 'rb') as file:
        binary_data = file.read()
        data_type = chardet.detect(binary_data)['encoding']
        if data_type==None :
            data = binary_data.decode('ISO-8859-1')
        else:
            data = binary_data.decode(data_type)
    return data

def precessing(input_data): #所有的预处理步骤
    ASCII_symbol = "!#$%&'()*+,-./0123456789:;<=>?@^\\_\"\n"
    remove_symbol = str.maketrans(ASCII_symbol, ' ' * len(ASCII_symbol))
    data = input_data.translate(remove_symbol)#取出数字、符号
    document = TextBlob(data)#分词
    splitword = document.words
    low_data = splitword.lower() #变为小写
    stem = WordNetLemmatizer()#词形还原
    steming_list = []
    for word in low_data:
        if not wordnet.synsets(word):#判是否是英语单词
            continue
        else:
            lemma = stem.lemmatize(word)
            lemma1 = stem.lemmatize(lemma, 'v')
            steming_list.append(lemma1)
    output_wordlist = [word for word in steming_list if not word in stopwords.words('english')]#去停用词
    return output_wordlist

def documents_read(input_path):
    '''
    读入所有数据，并得到经过预处理的训练和测试数据
    :param input_path: 数据集路径
    :return: 分别返回训练集和测试集的单词列表及文档分类标签
    '''
    root_catalogue = os.listdir(input_path) #root_catalogue为根目录下的文件夹列表_
    documents_wordlist = []     #存储训练集中所有文档的列表
    test_wordlist = []   #存储测试集中所有文档的列表
    train_class_label = []
    test_class_label = []
    label = 0
    for each_dirlist in root_catalogue:
        child_path = input_path + '\\' + each_dirlist + '\\' #child_path为子文件夹路径
        child_catalogue = os.listdir(child_path)  #child_catalogue为子文件夹目录，此处即为文档名称列表
        temp = 0
        label += 1
        for each_document in child_catalogue:
            each_documentpath = child_path + each_document  #所有文档的路径
            print(each_documentpath)
            document_data = open_file(each_documentpath)
            # 分类进行预处理
            temp = temp+1
            if temp % 5 == 0:
                test_class_label.append(label)
                test_wordlist.append(precessing(document_data))
            else:
                train_class_label.append(label)
                documents_wordlist.append(precessing(document_data))
    return documents_wordlist, test_wordlist,train_class_label,test_class_label

def documents_frequency(documents_wordlist):
    """
    计算所有单文档的单独词频
    :param documents_wordlist:list 是所有文档的词列表
    :return: list of collection.counter() 文档词频列表（一个字典的列表）
    """
    documents_frequency = []
    for document in documents_wordlist:
        documents_frequency.append(collections.Counter(document))
    return documents_frequency

def creat_dictionaries(dictionaries, start, end):
    """
    根据所有词频的词典，去掉高频和低频的键值项，生成新的词典
    :param dictionaries: （dic）
    :return:dictionaries: （dic）
    """
    for key, value in dictionaries.items():
        if value<start & value>end:
            del dictionaries[key]
    return dictionaries

def count_IDF(dictionary):
    """
    生成整个数据集的IDF
    :param vocabulary:dict of ‘词典’ = 词频
    :return: array of IDF
    """
    dictionary_freq = list(dictionary.values())
    freq_matrix = numpy.array(dictionary_freq)
    N = freq_matrix.sum()
    IDF_matrix = numpy.log(N/(freq_matrix+1))
    return IDF_matrix

def count_TF(document, dictionary):
    """
    计算一篇文档的词频
    :param document:collections.counter()
    :param dictionary: dict (words,Frequency)
    :return: array of TF
    """
    document_dict = dict(document.most_common())#先将collection.counter格式转换为字典
    keys = list(dictionary.keys())#取词典的单词
    document_count = {}    #保存输入文档按照总的词典的key统计得到的词频统计词典
    for key in keys:
        if document_dict.get(key):
            document_count[key] = document_dict[key]
        else:
            document_count[key] = 0
    vector = numpy.array(list(document_count.values()))
    TF = numpy.zeros(len(vector))
    for i in range(len(vector)):
        if vector[i] != 0:
            TF[i] = 1 + numpy.log(vector[i])
    return TF

def count_cosinvalue(x_train, x_test):
    """
    计算向量矩阵的cosin值
    :param x_test: 测试数据集M*K维的矩阵
           x_train: 训练数据集N*K维的矩阵
    :return: 返回 N * M 维的矩阵，表示第n个测试集数据到第m个训练集数据的cosine值。
    """
    inner_product = numpy.dot(x_test, x_train.T)
    norm_train = numpy.linalg.norm(x_train, ord=2, axis=1, keepdims=False)
    norm_test = numpy.linalg.norm(x_test, ord=2, axis=1, keepdims=False)
    norm_nd = numpy.dot(numpy.array([norm_test]).T, numpy.array([norm_train]))
    cosin_values = inner_product / norm_nd
    return numpy.nan_to_num(cosin_values)

def main():
    path = r'C:\file\20news-18828'
    documents_wordlist, test_wordlist, train_class_label, test_class_label = documents_read(path)  #读入所有文档并进行预处理后将结果存入documents_wordlist列表
    total_frequency = collections.Counter(list(_flatten(documents_wordlist)))   #去重，得到一个.counter（）类
    dictionary = dict(total_frequency.most_common())   #得到文档中所有词词频的词典
    for key, value in dictionary.items():#得到去除词频很高或很低的词典（最终使用的词典）
        if value<2 & value>5000:
            del dictionary[key]

    # 得到训练集和测试集的所有单文档的词频list，list中是每个文档的词频的counter类
    documentsfrequency = documents_frequency(documents_wordlist)
    test_set_frequency = documents_frequency(test_wordlist)

    #计算TF*IDF
    IDF = count_IDF(dictionary)
    VSM_list = []    #保存最终的VSM
    for document in documentsfrequency:
        TF = count_TF(document, dictionary)
        VSM = IDF * TF
        VSM_list.append(VSM)
    VSM_array = numpy.array(VSM_list)
    print('the type of train_VSM_array is ', VSM_array.shape)

    test_VSM_list = []
    for test_document in test_set_frequency:
        test_TF = count_TF(test_document, dictionary)
        test_VSM = IDF * test_TF
        test_VSM_list.append(test_VSM)
    test_VSM_array = numpy.array(test_VSM_list)
    print('the type of test_VSM_array is ', test_VSM_array.shape)

    #计算test set的cosin值(N*M矩阵)
    test_values = count_cosinvalue(VSM_array, test_VSM_array)
    print("N*M矩阵的维数：", test_values.shape)

    #K-NN方法
    K = 25
    correct_count = 0
    for i in range(len(test_class_label)):
        train_class_index = numpy.array(train_class_label)[test_values[i, :].argsort()]
        train_class_index = train_class_index.reshape(len(train_class_index), )
        train_class_index = train_class_index.tolist()
        test_list = train_class_index[len(train_class_label)-K-1: len(train_class_label)-1:] #取出矩阵的每一行元素值排序后的索引表
        test_class_count = collections.Counter(test_list)
        test_class = test_class_count.most_common(1)  #返回数量最多的元素及其个数
        predict_label = list(dict(test_class).keys())[0]
        if test_class_label[i] == predict_label:
            correct_count += 1
    correct_rate = correct_count/len(test_class_label)
    print("the correct rate of classifier is %.2f%%" % (correct_rate*100))
    return correct_rate

if __name__=='__main__':
    main()