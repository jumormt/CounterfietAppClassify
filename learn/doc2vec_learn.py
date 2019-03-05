from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# for i,doc in enumerate(common_texts):
#     print(doc)
#     print(i)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# todo:增量学习
# model.build_vocab(documents[3:],update=True)
#
# model.train(documents[3:], total_examples=model.corpus_count, epochs=model.iter)
model.infer_vector(['human'])
# print(model.docvecs.distance())
print(model.corpus_count)

print("end")