import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import LineSentence
import os
from sklearn.feature_extraction.text import TfidfVectorizer  # TFIDF
import codecs
from gensim import corpora, models, similarities

from common import calc_sim as cs

def tokenization(filename):
    result = []
    with open(filename, 'r') as f:
        text = LineSentence(f)
        for i in text:
            result.extend(i)
    return result


def main():
    dirpath = '..\\resource\\learn_data'

    corpus = []
    for dirpath, dirnames, filenames in os.walk(dirpath):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            corpus.append(tokenization(fullpath))
    dictionary = corpora.Dictionary(corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]# e.g.[[(1,2),(2,3)],[(1,2),(2,3)]]前为下标，后为词频

    tfidf = models.TfidfModel(doc_vectors)
    tfidfvectors = tfidf[doc_vectors]# e.g.[[(1,0.332),(2,0.12)],[(1,0.22),(2,0.45)]]前为下标，后为tf-idf值

    # test = tfidfvectors[0]
    # index = similarities.MatrixSimilarity(tfidfvectors)
    # sims = index[test] # 采用余弦相似度比较的方法，待比较向量：维度为dictionary的unique token的数量，将tfidf值填入对应下标，若无token则该下标处置0

    lsi = models.LsiModel(tfidfvectors, id2word=dictionary, num_topics=2)
    lsivectors = lsi[tfidfvectors]# e.g.[[(0,0.5),(1,0.7)],[(0,0.5),(1,0.7)]]前为主题下标，后为属于该主题的概率
    index = similarities.MatrixSimilarity(lsivectors)
    test = lsivectors[0]
    sims = index[test]
    # test1 = [i[1] for i in test]
    # test2 = lsivectors[2]
    # test21 = [i[1] for i in test2]
    # sim2 = cs.Cosine(test1, test21)

    lda = models.LdaModel(doc_vectors, id2word=dictionary, num_topics=2)# doc_Vectors也可以替换为tfidf的
    ldavectprs = lda[doc_vectors]# 同上
    index = similarities.MatrixSimilarity(ldavectprs)

    print(lda.print_topics(2)) # topic-word



    print("end")


if __name__ == '__main__':
    main()
