'''建立词汇表

每一个特征分别建立对应的词汇表

@author:chengxiao
@version:1.0
'''
__author__ = 'chengxiao'
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim.models import Word2Vec


def add_train(model, sentence):
    """继续添加词汇

    继续添加词汇，调用model.wv.vocab得到词汇表

    :param model: word2vec model
    :param sentence: add sentence
    :return: new model
    """

    model.build_vocab(sentence, update=True)
    model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)
    return model

def build_vocab(sentences):
    '''建立词汇表

    建立词汇表，调用model.wv.vocab得到词汇表

    :param sentences: 待建立词汇表的所有词汇
    :return: model
    '''

    model = Word2Vec(sentences, min_count=1)
    return model

def main():
    # 待建词汇表
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    model = build_vocab(sentences)
    print(model.wv.vocab['first'].index)
    # modelpath = 'test.model'
    # model = Word2Vec.load('test.model')
    # model.save(modelpath)

    addtr = [["hello", "world"]]
    model = add_train(model, addtr)
    print(model.wv.vocab)


if __name__ == '__main__':
    main()
