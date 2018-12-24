'''计算文本相似度

计算两个文本的相似度

@author:chengxiao
@version:1.0
'''

def calc_text_sim(res, file_name1, file_name2, model):
    '''计算两个文本相似度

    :param res: 总的文本词汇表
    :param file_name1: 待比较文本1的文件名
    :param file_name2: 待比较文本2的文件名
    :param model: 构建好的word2vec模型
    :return: 两个文本的相似度
    '''
    sentence1 = res[file_name1]
    sentence2 = res[file_name2]

    s1 = []
    s2 = []
    for i in sentence1:
        s1.extend(i)
    for i in sentence2:
        s2.extend(i)

    return model.wv.n_similarity(s1, s2)