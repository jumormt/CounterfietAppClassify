'''计算文本相似度

计算两个文本的相似度

@author:chengxiao
@version:1.0
'''
__author__ = 'chengxiao'
from common import calc_sim as cs


def list_add(mul1, list1, mul2, list2):
    '''
    两个列表权相加
    :param mul1:
    :param list1:
    :param mul2:
    :param list2:
    :return: "mul1*list1+mul2*list2"
    '''
    li = []
    for i in range(len(list1)):
        li.append(list1[i] * mul1 + list2[i] * mul2)
    return li

def calc_text_sim2(res, file_name1, file_name2, model):
    '''计算两个文本相似度

    :param res: 总的文本词汇表
    :param file_name1: 待比较文本1的文件名
    :param file_name2: 待比较文本2的文件名
    :param model: 构建好的word2vec模型
    :return: 两个文本的相似度
    '''
    file1bean = res[file_name1]
    file2bean = res[file_name2]
    if file1bean.tfidf_map == None:
        return file_name1 + " is empty"
    if file2bean.tfidf_map == None:
        return file_name2 + " is empty"

    s1 = [0] * model.wv.vector_size
    s2 = [0] * model.wv.vector_size
    count1 = 0
    for i in file1bean.tfidf_map.keys():
        count1 = count1 + file1bean.tfidf_map[i]
        s1 = list_add(file1bean.tfidf_map[i], list(model.wv[i]), 1, s1)
    s1 = [i / count1 for i in s1]

    count2 = 0
    for i in file2bean.tfidf_map.keys():
        count2 = count2 + file2bean.tfidf_map[i]
        s2 = list_add(file2bean.tfidf_map[i], list(model.wv[i]), 1, s2)
    s2 = [i / count2 for i in s2]
    return cs.Cosine(s1, s2)

def calc_text_sim(res, file_name1, file_name2, model):
    '''计算两个文本相似度

    :param res: 总的文本词汇表
    :param file_name1: 待比较文本1的文件名
    :param file_name2: 待比较文本2的文件名
    :param model: 构建好的word2vec模型
    :return: 两个文本的相似度
    '''
    file1bean = res[file_name1]
    file2bean = res[file_name2]

    s1 = [0] * model.wv.vector_size
    s2 = [0] * model.wv.vector_size
    count1 = 0
    for j in file1bean.sentences:
        for i in j:
            count1 = count1 + file1bean.tfidf_map[i]
            s1 = list_add(file1bean.tfidf_map[i], list(model.wv[i]), 1, s1)
    s1 = [i / count1 for i in s1]

    count2 = 0
    for j in file2bean.sentences:
        for i in j:
            count2 = count2 + file2bean.tfidf_map[i]
            s2 = list_add(file2bean.tfidf_map[i], list(model.wv[i]), 1, s2)
    s2 = [i / count2 for i in s2]
    return cs.Cosine(s1, s2)

def calc_text_sim_old(res, file_name1, file_name2, model):
    '''计算两个文本相似度(未采用tfidf)

    :param res: 总的文本词汇表
    :param file_name1: 待比较文本1的文件名
    :param file_name2: 待比较文本2的文件名
    :param model: 构建好的word2vec模型
    :return: 两个文本的相似度
    '''
    file1bean = res[file_name1]
    file2bean = res[file_name2]
    if file1bean.tfidf_map == None:
        return file_name1 + " is empty"
    if file2bean.tfidf_map == None:
        return file_name2 + " is empty"

    s1 = []
    s2 = []
    for i in file1bean.sentences:
        s1.extend(i)
    for i in file2bean.sentences:
        s2.extend(i)

    return model.wv.n_similarity(s1, s2)

def calc_str_sim(file_path1, file_path2):
    '''
    计算两个文件的编辑距离
    :param file_path1:
    :param file_path2:
    :return: 两个文件字符串的相似度
    '''

    f1 = open(file_path1, 'r', encoding='utf-8')
    f2 = open(file_path2, 'r', encoding='utf-8')
    str1 = f1.read()
    str2 = f2.read()
    if len(str1) == 0:
        f1.close()
        f2.close()
        return file_path1+' is empty'
    if len(str2) == 0:
        f1.close()
        f2.close()
        return file_path2+' is empty'
    sim = cs.Edit_distance_str(str1, str2)['Similarity']
    f1.close()
    f2.close()
    return sim
