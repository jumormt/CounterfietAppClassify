'''构建目录下文本词汇表

构建目录下所有文件的文本词汇表，返回一个以文本名为键的map,value为TextBean

@author:chengxiao
@version:1.0
'''
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import LineSentence
import os

import build_vocab as bv
import calc_sim as cs
import calc_textsim as cts
from TextBean import TextBean

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
            files, model = build_text(files, tb, model)
    return files, model

def main():
    dirpath = 'testdata'
    files, model = build_vocab_all(dirpath)
    print(cts.calc_text_sim(files, 'testdata.txt', 'testdata2.txt', model))


if __name__ == '__main__':
    main()

    print("end")
