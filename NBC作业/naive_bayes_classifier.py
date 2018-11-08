'''
80%训练集，20%测试集
'''
import os
import json
import chardet  #用于操作文档编码格式
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter
import math

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

    train_class = []
    test_class = []
    for each_dirlist in root_catalogue:
        child_path = input_path + '\\' + each_dirlist + '\\' #child_path为子文件夹路径
        child_catalogue = os.listdir(child_path)  #child_catalogue为子文件夹目录，此处即为文档名称列表
        temp = 0
        document_wordlist = []
        test_wordlist = []
        for each_document in child_catalogue:
            each_documentpath = child_path + each_document  #所有文档的路径
            print(each_documentpath)
            document_data = open_file(each_documentpath)
            # 分类进行预处理
            temp = temp+1
            if temp % 5 == 0:
                test_wordlist.append(precessing(document_data))
            else:
                document_wordlist.append(precessing(document_data))
        train_class.append(document_wordlist)
        test_class.append(test_wordlist)

        print(len(train_class))
        print(len(test_class))
    return train_class, test_class

def classify(train_class, test_class):
    '''

    :param train_class: 经过预处理的训练集
    :param test_class: 经过预处理的测试集
    :return: 返回分类正确率
    '''
    # 需要用到的数据结构
    train_dict_list = []  # the list of word frequency dictionary of every document  in training set [{}...{}{}]
    test_dict_list = []  # the list of word frequency dictionary of every document  in testing set [{}...{}{}]
    class_word_num = []  # the list of the number of the word in every training set class
    class_document_num = []  # the list of the number of document in every training set class
    all_words_num = 0  # the number of all words in training set
    all_document_num = 0  # the number of all documents in training set
    # 将数据存入上述的数据结构中
    for i in range(len(train_class)):
        class_document_num.append(len(train_class[i]))
        all_document_num += len(train_class[i])
        document_word_num = 0
        temp_class_word = []
        temp_class = []
        for j in range(len(train_class[i])):
            document_word_num += len(train_class[i][j])
            all_words_num += len(train_class[i][j])
            temp_class_word.extend(train_class[i][j])
        train_temp = Counter(temp_class_word)
        train_dict_list.append(dict(train_temp.most_common()))
        class_word_num.append(document_word_num)
        for k in range(len(test_class[i])):
            test_temp = Counter(test_class[i][k])
            temp_class.append(dict(test_temp.most_common()))
        test_dict_list.append(temp_class)
    # 下面两行将最终的训练集和测试集数据结构化数据写入文档，供调试时检查
    #write_dict(r'D:\python_files\python_projects\data_mining\NBC\train_dict_list.txt', train_dict_list)
    #write_dict(r'D:\python_files\python_projects\data_mining\NBC\test_dict_list.txt', test_dict_list)

    correct_num = 0
    test_num = 0
    for i in range(len(test_dict_list)):
        for j in range(len(test_dict_list[i])):
            test_num += 1
            predict = []
            for k in range(len(train_dict_list)):
                class_p = 0
                class_p1 = 0
                for key in test_dict_list[i][j].keys():
                    #计算有一个词在一个类中的概率log值
                    if key not in train_dict_list[k].keys():
                        continue
                    else:
                        value = train_dict_list[k][key]
                        temp_p = math.log(value / class_word_num[k])
                        class_p += temp_p
                    #计算一个词在另外19个类中的概率log值
                    value1 = 0
                    for m in range(len(train_dict_list)):
                        if m == k:
                            continue
                        else:
                            if key not in train_dict_list[m].keys():
                                continue
                            else:
                                value1 += train_dict_list[m][key]
                    if value1 == 0:
                        temp_p1 = 0
                    else:
                        temp_p1 = math.log(value1 / (all_words_num - class_word_num[k]))
                    class_p1 += temp_p1
                #计算一个类中的文档在该类中的概率log值加上该文档所有词出现在该类中概率log值的和
                class_p += math.log(class_document_num[k] / all_document_num)
                class_p1 += math.log((all_document_num - class_document_num[k]) / all_document_num)
                #计算最终的概率值
                predict.append(class_p / class_p1)
            #判断正确的文档数
            if predict.index(min(predict)) == i:
                correct_num += 1
    predict_rate = correct_num / test_num
    return predict_rate

def main():
    #下面四行是数据集的读取分类及预处理过程，为提高调试效率，后面直接从文件中读取已经处理好的数据
    #path = r'D:\file\20news-18828'
    #train_class, test_class = documents_read(path)
    #write_dict(r'D:\python_files\python_projects\data_mining\NBC\train_class.txt', train_class)
    #write_dict(r'D:\python_files\python_projects\data_mining\NBC\test_class.txt', test_class)

    #读出数据
    train_class = read_dict(r'D:\python_files\python_projects\data_mining\NBC\train_class.txt')
    test_class = read_dict(r'D:\python_files\python_projects\data_mining\NBC\test_class.txt')
    #计算正确率
    predict_rate = classify(train_class, test_class)
    print("The correct rate is %.2f%%" % (predict_rate * 100))

if __name__=='__main__':
    main()