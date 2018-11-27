import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary

train = []
stopwords = codecs.open('stopwords.txt', 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
fp = codecs.open('wiki.zh.seg.utf.txt', 'r', encoding='utf8')
for line in fp:
    line = line.split()
    train.append([w for w in line if w not in stopwords])

dictionary = Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
