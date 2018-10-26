'''
向量空间模型
'''
import os
import json
import numpy
import chardet  #用于操作文档编码格式
from textblob import TextBlob
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import digits #从字符串中去除数字res = s.translate(None, digits)
from tkinter import _flatten
#用于拉直文档列表
import collections
import nltk

def open_file(path):
    '''
    读取文件内容,返回存储文档的字符串
    :parameter:path（文档路径）
    :return:data(文档字符串)
    '''
    document_list = []
    with open(path, 'rb') as file:
        binary_data = file.read()
        data_type = chardet.detect(binary_data)['encoding']
        if data_type==None :
            data = binary_data.decode('ISO-8859-1')
        else:
            data = binary_data.decode(data_type)
        #print(data)
    return data

def document_precess(input_data):
    '''
    第1次预处理：
    处理掉数字和所有符号、将字符串变成全部小写
    :param input_data:（string）
    :return:output_data（string）(去除除空格以外的数字和符号的小写英文字符串)
    '''
    #去掉数字和符号（用空格代替）
    ASCII_symbol = "!#$%&'()*+,-./0123456789:;<=>?@\"\n"
    remove_symbol = str.maketrans(ASCII_symbol, ' '*len(ASCII_symbol))
    output_data = input_data.translate(remove_symbol)
    #去掉多余的空格
    #output_data = output_data.replace('  ', ' ')
    #将多有的字母变成小写
    output_data = output_data.lower()
    return output_data

def split_word(input_data):
    '''
    将输入的字符串分成词存入列表中
    :param input_data: （string）
    :return: output_list:(list)
    '''
    #也可用output_list = input_data.split()
    document = TextBlob(input_data)
    output_list = document.words
    return output_list

def steming(input_list):
    '''
    将输入的列表中的词进行词干提取后输出一个词干提取后的词列表
    :param input_list:（list）
    :return:out_list:(list)
    '''
    stem = SnowballStemmer('english')
    steming_list = []
    for word in input_list:
        steming_list.append(stem.stem(word))
    return steming_list

def remove_stopword(input_list):
    '''
    去除停用词
    :param input_list: （list）
    :return: output_list:(list)
    '''
    output_list = [word for word in input_list if not word in stopwords.words('english')]
    return output_list

def precessing(input_data): #所有的预处理步骤
    precess_data = document_precess(input_data)
    split_wordlist = split_word(precess_data)
    steming_list = steming(split_wordlist)
    output_wordlist = remove_stopword(steming_list)
    return output_wordlist

#将文件目录中的所有文档经过预处理后读入一个documents_wordlist中
def documents_read(input_path):
    root_catalogue = os.listdir(input_path) #root_catalogue为根目录下的文件夹列表_
    documents_wordlist = []     #存储训练集中所有文档的列表
    test_wordlist = []   #存储测试集中所有文档的列表
    test_label_list = []
    test_lable = 0  #0表示训练数据，1表示测试数据
    for each_dirlist in root_catalogue:
        child_path = input_path + '\\' + each_dirlist + '\\' #child_path为子文件夹路径
        child_catalogue = os.listdir(child_path)  #child_catalogue为子文件夹目录，此处即为文档名称列表
        temp = 0
        for each_document in child_catalogue:
            each_documentpath = child_path + each_document  #所有文档的路径
            print(each_documentpath)
            document_data = open_file(each_documentpath)
            # 进行预处理，并将每个文档预处理得到的列表内容加到documents_wordlist中
            temp = temp+1
            if temp%5 == 0:
                test_lable = 1
                test_wordlist.append(precessing(document_data))
            else:
                documents_wordlist.append(precessing(document_data))
                test_lable = 0
            test_label_list.append(test_lable)
    write_file(r'C:\file\datamining_temp\test_wordlist.txt', json.dumps(test_wordlist, ensure_ascii=False))
    write_file(r'C:\file\datamining_temp\test_label_list.txt', json.dumps(test_label_list, ensure_ascii=False))
    return documents_wordlist


def write_file(path, data):
    '''
    :param path: 即将写入文件的路径
    :param data: 即将写入文件的内容
    :return: None
    '''
    with open(path, 'w', errors='ignore') as file:
        file.write(data)

#从文件中读出字典
def read_dict(path):
    with open(path, 'r') as f:
        dict = eval(f.read())
    return dict

#将字典写入文件
def write_dict(path, dict):
    json_dict = json.dumps(dict)
    write_file(path, json_dict)

#为所有的文档设置文档分类标签
def document_label_set(input_path):
    file_path = r'C:\file\datamining_temp\documents_labellist.txt'
    root_catalogue = os.listdir(input_path) #root_catalogue为根目录下的文件夹列表_
    documents_labellist = []     #存储所有文档中词的列表
    label = 0
    for each_dirlist in root_catalogue:
        child_path = input_path + '\\' + each_dirlist + '\\' #child_path为子文件夹路径
        child_catalogue = os.listdir(child_path)  #child_catalogue为子文件夹目录，此处即为文档名称列表
        label += 1
        for each_document in child_catalogue:
            documents_labellist.append(label)
    label_json = json.dumps(documents_labellist, ensure_ascii=False)
    write_file(file_path, label_json)
    return documents_labellist

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
    :return:ictionaries: （dic）
    """
    #dictionaries = dict(documents_frequency.most_common())
    dictionaries_name = list(dictionaries.keys())
    dictionaries_count = list(dictionaries.values())#--------------------------------------------------------------------
    dictionaries = {}
    dictionaries_name = dictionaries_name[start:end:1]
    dictionaries_count = dictionaries_count[start:end:1]
    for i in range(len(dictionaries_name)):
        dictionaries[dictionaries_name[i]] = dictionaries_count[i]
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
    IDF_matrix.tofile(r"C:\file\datamining_temp\IDF.txt")
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
    #print(Document_count) #打印词典统计
    vector = numpy.array(list(document_count.values()))
    vector_len = len(vector)
    TF = numpy.zeros(vector_len)
    for i in range(vector_len):
        if vector[i] != 0:
            TF[i] = 1 + numpy.log(vector[i])
    return TF

def main():
    path = r'C:\file\20news-18828'
    documents_wordlist = documents_read(path)  #读入所有文档并进行预处理后将结果存入documents_wordlist列表
    #print(documents_wordlist)
    #document_label_set(path)
    print(len(documents_wordlist))


    all_wordslist = list(_flatten(documents_wordlist))  #将得到的所有词的列表变成一维的列表

    print(len(all_wordslist))
    total_frequency = collections.Counter(all_wordslist)   #去重，得到一个.counter（）类
    dictionary = dict(total_frequency.most_common())   #得到文档中所有词词频的词典

    write_dict(r'C:\file\datamining_temp\final_dictionary.txt', dictionary)

    final_dictionary = creat_dictionaries(dictionary, 500, 18000)  #得到去除词频很高或很低的词典（最终使用的词典）

    #输出最终的词典
    json_dictionary = json.dumps(final_dictionary, ensure_ascii=False)
    f = open(r'C:\file\datamining_temp\final_dictionary.txt', 'wb')
    f.write(json_dictionary.encode('utf-8'))
    f.close()

    # 得到所有单文档的词频list，list中是每个文档的词频的counter类
    documentsfrequency = documents_frequency(documents_wordlist)


    #计算TF和IDF
    IDF = count_IDF(final_dictionary)
    #下面的document为每篇文章中的词频
    TF_list = []  #保存所有文档的TF
    VSM_list = []    #保存最终的VSM
    for document in documentsfrequency:
        TF = count_TF(document, final_dictionary)
        TF_list.append(TF)
        VSM = IDF * TF
        #print(VSM)
        VSM_list.append(VSM)
    TF_array = numpy.array(TF_list)
    print("the typeo of TF_array is ", TF_array.shape)
    TF_array.tofile(r"C:\file\datamining_temp\TF.txt")
    VSM_array = numpy.array(VSM_list)
    print('the type of VSM_array is ', VSM_array.shape)
    VSM_array.tofile(r'C:\file\datamining_temp\VSM.txt')
    return VSM_array

if __name__=='__main__':
    main()