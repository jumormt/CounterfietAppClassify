'''文本预处理

对文本进行去分词

@author:chengxiao
@version:1.0
'''
__author__ = 'chengxiao'
import os

from sklearn.feature_extraction.text import TfidfVectorizer #TFIDF


def TFIDF(word_lemmatized_stemmered_wordonly):
    # TFIDF计算
    tf_idf = TfidfVectorizer()  # 初始化对象
    tf_data = tf_idf.fit_transform(word_lemmatized_stemmered_wordonly)  # 计算TFIDF值
    words = tf_idf.get_feature_names()  # 取出所统计单词项
    TFIDF = dict()  # 创建空字典
    for i in range(len(word_lemmatized_stemmered_wordonly)):
        for j in range(len(words)):

            TFIDF[words[j]] = tf_data[i, j]
    return TFIDF
    # path_TFIDF = 'resource\\testdata\\words_atheism_word_tokenize_Stopped_postag_lemmatized_stemmered_TFIDF2.txt'
    # path_TFIDF_sorted = 'resource\\testdata\\words_atheism_word_tokenize_Stopped_postag_lemmatized_stemmered_TFIDF_sorted2.txt'

    # with open(path_TFIDF, 'w', encoding='utf-8') as f:
    #     # 向文件写入TFIDF值
    #     for i in range(len(word_lemmatized_stemmered_wordonly)):
    #         for j in range(len(words)):
    #             if tf_data[i, j] > 1e-5:
    #                 f.write(words[j] + ':' + str(tf_data[i, j]))
    #                 f.write('\n')
    #                 TFIDF[str(words[j])] = tf_data[i, j]
    #     print("TFIDF written.")


    # TFIDFSorted = sorted(TFIDF.items(), key=lambda e: e[1], reverse=True)
    # # 按TFIDF值大小排序
    #
    # with open(path_TFIDF_sorted, 'w', encoding='utf-8') as f:
    #     # 向文件写入排序后的TFIDF值
    #     for key in TFIDFSorted:
    #         f.write(str(key))
    #         f.write('\n')
    # print("TFIDF sorted written.")


if __name__ == '__main__':
    # dir = 'resource\\testdata'
    # text_preprocess_all(dir)
    dirnew = '..\\resource\\testdata_new'
    for dirpath, dirnames, filenames in os.walk(dirnew):

        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            fullpath_l = []
            with open(fullpath, 'r') as f:
                fullpath_l.append(f.read())
            tfidf = TFIDF(fullpath_l)
            print(sum(tfidf.values()))