'''计算相似度

计算两个向量文本的相似度

'''
import numpy as np
import math


def Edit_distance_str(str1, str2):
    '''编辑距离

    编辑距离，用于计算两个字符串之间的编辑距离，传入参数为两个字符串

    :param str1:
    :param str2:
    :return:
    '''
    import Levenshtein
    edit_distance_distance = Levenshtein.distance(str1, str2)
    similarity = 1 - (edit_distance_distance / max(len(str1), len(str2)))
    return {'Distance': edit_distance_distance, 'Similarity': similarity}


def Edit_distance_array(str_ary1, str_ary2):
    '''针对列表改写的编辑距离

    针对列表改写的编辑距离，在NLP领域中，计算两个文本的相似度，是基于句子中词和词之间的差异。
    如果使用传统的编辑距离算法，则计算的为文本中字与字之间的编辑次数。这里根据编辑距离的思维，
    将编辑距离中的处理字符串中的字符对象，变成处理list中每个元素

    :param str_ary1:
    :param str_ary2:
    :return:
    '''
    len_str_ary1 = len(str_ary1) + 1
    len_str_ary2 = len(str_ary2) + 1
    matrix = [0 for n in range(len_str_ary1 * len_str_ary2)]
    for i in range(len_str_ary1):
        matrix[i] = i
    for j in range(0, len(matrix), len_str_ary1):
        if j % len_str_ary1 == 0:
            matrix[j] = j // len_str_ary1
    for i in range(1, len_str_ary1):
        for j in range(1, len_str_ary2):
            if str_ary1[i - 1] == str_ary2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str_ary1 + i] = min(matrix[(j - 1) * len_str_ary1 + i] + 1,
                                               matrix[j * len_str_ary1 + (i - 1)] + 1,
                                               matrix[(j - 1) * len_str_ary1 + (i - 1)] + cost)
    distance = int(matrix[-1])
    similarity = 1 - int(matrix[-1]) / max(len(str_ary1), len(str_ary2))
    return {'Distance': distance, 'Similarity': similarity}


def Cosine(vec1, vec2):
    '''余弦相似度

    Cosine，余弦夹角
    机器学习中借用这一概念来衡量样本向量之间的差异。也可以使用在余弦相似度算法中

    :param vec1:
    :param vec2:
    :return:
    '''
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2) / (math.sqrt((npvec1 ** 2).sum()) * math.sqrt((npvec2 ** 2).sum()))

def Euclidean(vec1, vec2):
    '''欧式距离

    :param vec1:
    :param vec2:
    :return: 欧式距离
    '''
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())


def Manhattan(vec1, vec2):
    '''曼哈顿距离

    :param vec1:
    :param vec2:
    :return:
    '''
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()


def Chebyshev(vec1, vec2):
    '''切比雪夫距离

    :param vec1:
    :param vec2:
    :return:
    '''
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return max(np.abs(npvec1-npvec2))


def Mahalanobis(vec1, vec2):
    '''马氏距离

    :param vec1:
    :param vec2:
    :return:
    '''
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    npvec = np.array([npvec1, npvec2])
    sub = npvec.T[0]-npvec.T[1]
    inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))
