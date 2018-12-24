'''文本信息的封装

@author:chengxiao
@version:1.0
'''
class TextBean(object):
    '''文本信息的封装

    文本信息的封装

    Attributes:
        __sentences : 文本分词后的向量集合，e.g.[['first', 'sentence', '.'], ['second', 'sentence', '.']]
        __file_path : 文本相对路径
        __file_name : 文本文件名
    '''

    def __init__(self, file_name=None, sentences=None, file_path=None):
        '''
        constructor

        :param file_name:
        :param sentences:
        :param file_path:
        '''
        self.__sentences = sentences
        self.__file_path = file_path
        self.__file_name = file_name

    @property
    def sentences(self):
        '''

        :return:
        '''
        return self.__sentences

    @sentences.setter
    def sentences(self, value):
        self.__sentences = value

    @property
    def file_path(self):
        '''

        :return:
        '''
        return self.__file_path

    @file_path.setter
    def file_path(self, value):
        self.__file_path = value

    @property
    def file_name(self):
        '''

        :return:
        '''
        return self.__file_name

    @file_name.setter
    def file_name(self, value):
        self.__file_name = value