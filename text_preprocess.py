'''文本预处理

对文本进行词性还原、分句分词去停用词处理

@author:chengxiao
@version:1.0
'''
import os
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer #词性还原
from nltk.tokenize import sent_tokenize #分句
from nltk.tokenize import word_tokenize #分词
from nltk.corpus import stopwords       #去停用词
from nltk.stem import SnowballStemmer   #词干提取
from sklearn.feature_extraction.text import TfidfVectorizer #TFIDF
import chardet                          #检测编码格式
import re    #匹配去标点符号，特殊字符

cachedStopWords = stopwords.words("english")

def word_tokenize_stopwords_removal(text):
    '''对整个文本进行分词，这里为不分句直接分词,并去停用词、标点、特殊字符、带符号单词

    :param text: 待处理文本
    :return: 处理后的文本
    '''

    # atheism = re.sub("[+:\.\!\/_,$%^*(+\"\'<>=]+|[+——！，。？、~@#￥%……&*（）]+", " ", atheism)
    # words = word_tokenize(atheism)
    # 分词前去掉符号标点和特殊字符，转化为空格，也可以先分词再去掉含标点的词，后者去掉的东西更多，这里采取后一种

    # 分词，同时直接去掉所有带符号的词，如邮箱后缀、hyphen连词、缩写等
    words = [word for word in word_tokenize(text) if (str.isalpha(word) is not False)]
    print('去掉所有带符号的词，如邮箱后缀、hyphen连词、缩写.')

    words = [word for word in word_tokenize(text) if (word.encode( 'UTF-8' ).isalpha())]
    print('去掉非英文')

    # 小写化后去停用词+去长度小于3的单词+去数字和包含符号的单词如 2-year
    word_stopped = [w.lower() for w in words if (w.lower() not in cachedStopWords and len(w) > 2 and str.isalpha(w) is not False)]
    print('小写化后去停用词+去长度小于3的单词+去数字和包含符号的单词.')

    return word_stopped

def word_pos_tags(word_stopped):
    '''词性标注

    :param word_stopped:
    :return: 单词+词性标注为元组的list: pos_tags
    '''

    pos_tags = nltk.pos_tag(word_stopped)
    print('Pos_tags written.')
    return pos_tags

def get_wordnet_pos(treebank_tag):
    '''词性标注提取

    :param treebank_tag:
    :return:
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_string(pos_tags):
    '''词形还原后词干提取函数

    :param pos_tags:
    :return: 还原后的单词列表
    '''

    res = []
    lemmatizer = WordNetLemmatizer()  # 初始化词形还原对象
    stemmer = SnowballStemmer("english")  # 选择语言，初始化词干提取对象
    for word, pos in pos_tags:
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(stemmer.stem(lemmatizer.lemmatize(word, pos=wordnet_pos)))
    return res


def text_preprocess(file_path, outfile_path):
    '''文本预处理

    对文本进行词性还原、分句分词去停用词处理

    :param file_path: 待处理文本路径
    :param outfile_path: 处理后输出文本路径
    :return: 处理后的单词列表
    '''
    # 进行词形还原和词干提取,并输出记录结果
    # 返回仅由空格分隔单词的纯文本，即一个string的list: wordLemmatizedStemmeredWordOnly
    with open(file_path, 'r', encoding='utf-8') as f:
        print("---------------"+"start!processing"+file_path)
        text = f.read()
        stop_words = word_tokenize_stopwords_removal(text)
        pos_tags_words = word_pos_tags(stop_words)
        word_lemmatized_stemmered = lemmatize_string(pos_tags_words)


        with open(outfile_path, 'w', encoding='utf-8') as f:
            for word in word_lemmatized_stemmered:
                f.write(str(word))
                f.write(str(' '))
        print("---------------"+"done!processing"+file_path)
        return word_lemmatized_stemmered

def text_preprocess_all(dir_path):
    '''
    文件夹内文本预处理，处理后的文本放置在同级目录下新建_new文件夹下
    :param dir_path:
    :return:
    '''
    for dirpath, dirnames, filenames in os.walk(dir_path):
        output_dir = dirpath + '_new'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            out_filepath = os.path.join(output_dir, file)
            text_preprocess(fullpath, out_filepath)


if __name__ == '__main__':
    # filepath = 'resource\\testdata\\testdata3'
    # out_filepath = 'resource\\testdata\\testdata3_tmp2'
    # text_preprocess(filepath, out_filepath)
    dir = 'resource\\testdata'
    text_preprocess_all(dir)