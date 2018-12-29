'''
main
'''
__author__ = 'chengxiao'
from text_analysis import build_vocab_all as bva, calc_textsim as cts
import os
from common import calc_sim as cs
import json
from common import build_vocab as bv


def get_json_sim(json_dirpath):
    '''

    :param json_dirpath:
    :return: key为pair对，vaule为以各个比较项为键值的相似度
    '''
    files = {}
    for dirpath, dirnames, filenames in os.walk(json_dirpath):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            with open(fullpath, 'r', encoding='utf-8') as f:
                fs = json.load(f)
                files[file] = {'json': fs, 'filepath': fullpath}
    models = dict()

    sentences = []
    for key in files:
        fj = files[key]['json']['details']['app_details']['app_category']
        sentences.append(fj)
    models['app_category'] = bv.build_vocab(sentences)
    sentences = []

    sentences = []
    for key in files:
        fj = files[key]['json']['details']['app_details']['app_type']
        sentences.append([fj])
    models['app_type'] = bv.build_vocab(sentences)
    sentences = []

    for key in files:
        appcategory = files[key]['json']['details']['app_details']['app_category'][0]
        app_type = files[key]['json']['details']['app_details']['app_type']

        appcategory_vocab = models['app_category'].wv.vocab
        app_type_vocab = models['app_type'].wv.vocab
        files[key]['vec'] = []
        files[key]['vec'].append(int(files[key]['json']['doc_type']))
        files[key]['vec'].append(int(files[key]['json']['backend_id']))

        files[key]['vec'].append(appcategory_vocab[appcategory].index/len(appcategory_vocab))
        files[key]['vec'].append(app_type_vocab[app_type].index/len(app_type_vocab))
        print(files[key]['vec'])

    result = dict()
    texts = list(files.keys())
    for i in range(len(texts) - 1):
        for j in range(i + 1, len(texts)):
            file1 = texts[i]
            file2 = texts[j]
            reskey = tuple([file1, file2])
            result[reskey] = dict()
            simm = cs.Cosine(files[file1]['vec'], files[file2]['vec'])
            result[reskey]['json'] = simm

    return result



def get_text_sim(long_des_dir, short_des_dir):
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
    json_dirpath = 'resource\\testdata_json\\'


    text_result = get_text_sim(long_des_dir, short_des_dir)
    json_result = get_json_sim(json_dirpath)


if __name__ == '__main__':
    main()
