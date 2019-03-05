from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# for i,doc in enumerate(common_texts):
#     print(doc)
#     print(i)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# print(model.docvecs.most_similar('0'))
# # 进行相关性比较
# print(model.docvecs.similarity())
# # 输出标签为‘10’句子的向量
# print(model.docvecs['10'])
# # 也可以推断一个句向量(未出现在语料中)
# words = u"여기 나오는 팀 다 가슴"
# print(model.infer_vector(words.split()))
# # 也可以输出词向量
# print(model[u'가슴'])
model.infer_vector(['human'])

print("end")