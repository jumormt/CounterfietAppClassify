from text_analysis import build_vocab_all as bva, calc_textsim as cts
import os
from common import calc_sim as cs
__author__ = 'chengxiao'

def process(long_des_dir, short_des_dir):
    '''

    :param long_des_dir: 长描述文件夹（不含子目录）
    :param short_des_dir: 短文本文件夹
    :return: key为pair对，vaule为以各个比较项为键值的相似度
    '''

    dirdic = dict() # 长描述文本，用于存放files和model
    # dirpath = 'resource\\testdata2_new'
    # files, model = bva.build_vocab_all(dirpath)
    # texts = list(files.keys())
    # dirdic['testdata2'] = [files, model]
    # dirpath = 'resource\\extractdata\\txtExtract\\'

    files, model = bva.build_vocab_all(long_des_dir)
    texts = list(files.keys())
    dirdic['longdes'] = {'files': files, 'model': model}

    result = dict()

    attrs = os.listdir(short_des_dir)
    # texts = []
    # for dirpath, dirnames, filenames in os.walk(short_des_dir+attrs[0]):
    #     for file in filenames:
    #         texts.append(file)

    for i in range(len(texts) - 1):
        for j in range(i + 1, len(texts)):
            file1 = texts[i]
            file2 = texts[j]
            reskey = tuple([file1, file2])
            result[reskey] = dict()

            # 长文本的相似度
            for attr in dirdic:
                simm = cts.calc_text_sim_old(dirdic[attr]['files'], file1, file2, dirdic[attr]['model'])
                result[reskey][attr] = simm
                print(file1, ",", file2, "in", attr, " sim:", simm)
            # 短文本的相似度
            for attr in attrs:
                file1path = short_des_dir + attr + "\\" + file1
                file2path = short_des_dir + attr + "\\" + file2
                simm = cts.calc_str_sim(file1path, file2path)
                print(file1, ",", file2, "in", attr, " sim:", simm)
                result[reskey][attr] = simm
    print("end")
    return result

def main():
    long_des_dir = 'resource\\testdata3_long\\'
    short_des_dir = 'resource\\testdata3_short\\'

    process(long_des_dir, short_des_dir)

if __name__ == '__main__':
    main()
