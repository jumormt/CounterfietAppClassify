from gensim.models import Word2Vec

def addTrain(model, sentence):
    model.build_vocab(sentence, update=True)
    model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)

sentences = [['first', 'sentence'], ['second', 'sentence']]
model = Word2Vec(sentences, min_count=1)

print(model.wv.vocab['first'].index)
# modelpath = 'test.model'
# model = Word2Vec.load('test.model')
# model.save(modelpath)

addtr = [["hello", "world"]]
addTrain(model, addtr)
print(model.wv.vocab)