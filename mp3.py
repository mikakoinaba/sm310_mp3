import nltk
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

# read in fake data and split into words
fake = open('clean_fake.txt', encoding='utf-8').read()

fake_sentences = fake.split('\n')

fake_words = []
for title in fake_sentences:
	if title != '':
		fake_words.extend(title.split(' '))


# read in real data and split into words
real = open('clean_real.txt', encoding='utf-8').read()

real_sentences = real.split('\n')

real_words = []
for title in real_sentences:
	if title != '':
		real_words.extend(title.split(' '))


# Part 1
def sortTuple(tup):    
    return(sorted(tup, key = lambda x: x[1], reverse=True))

wordCountFake = []
for word in set(fake_words):
	if word not in stopWords:
		wordCountFake.append((word, fake_words.count(word)))

wordCountReal = []
for word in set(real_words):
	if word not in stopWords:
		wordCountReal.append((word, real_words.count(word)))


print(sortTuple(wordCountFake)[0:10])
print(sortTuple(wordCountReal)[0:10])

# Top ten most seen words (% of text)
# num of different words
# num times we see clinton vs trump