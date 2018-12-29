'''构建目录下文本词汇表

构建目录下所有文件的文本词汇表，返回一个以文本名为键的map,value为TextBean

@author:chengxiao
@version:1.0
'''
__author__ = 'chengxiao'
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import LineSentence
import os
from sklearn.feature_extraction.text import TfidfVectorizer #TFIDF

from common import build_vocab as bv
from text_analysis.TextBean import TextBean

def add_TFIDF_map(textbean):
    '''增加tfidf字段

    给文本增加词权重字段

    :param textbean: 该文本的bean
    :return: 增加字段后的bean
    '''
    with open(textbean.file_path, 'r') as f:
        lis = []
        ff = f.read()
        if not ff: # 文件为空
            return textbean
        lis.append(ff)
        # TFIDF计算
        tf_idf = TfidfVectorizer()  # 初始化对象
        tf_data = tf_idf.fit_transform(lis)  # 计算TFIDF值
        words = tf_idf.get_feature_names()  # 取出所统计单词项
        TFIDF = dict()  # 创建空字典
        for i in range(len(lis)):
            for j in range(len(words)):

                TFIDF[words[j]] = tf_data[i, j]
        textbean.tfidf_map = TFIDF
        return textbean

def build_text(res, textbean, model=None):
    '''构建单个文本的文本词汇表

    构建单个文本的文本词汇表

    :param res: 添加该文本词汇表后的文本词汇表
    :param textbean: 该文本的bean
    :param model: 添加该文本前的word2vec模型
    :return: 文本词汇表，添加该文本后的word2vec模型
    '''
    with open(textbean.file_path, 'r') as f:
        sentences = LineSentence(f)
        if model:
            model = bv.add_train(model, sentences)
        else:
            model = bv.build_vocab(sentences)
        li = []
        for st in sentences:
            li.append(st)
        textbean.sentences = li
        res[textbean.file_name] = textbean
    return res, model

def build_vocab_all(dir_path):
    '''构建目录下的文本词汇表

    :param dir_path: 目录路径
    :return: 总的文本词汇表，总的word2vec模型
    '''

    files = {}
    model = None
    for dirpath, dirnames, filenames in os.walk(dir_path):

        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            tb = TextBean(file_name=file, file_path=fullpath)
            tb = add_TFIDF_map(tb)
            files, model = build_text(files, tb, model)
    return files, model

def main():
    dirpath = '..\\resource\\testdata_new'
    files, model = build_vocab_all(dirpath)
    # print(cts.calc_text_sim(files, 'testdata.txt', 'testdata2.txt', model))
    print('end')


if __name__ == '__main__':
    main()
