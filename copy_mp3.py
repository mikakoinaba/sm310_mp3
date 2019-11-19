
import nltk
# from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn
import numpy as np

# stopWords = set(stopwords.words('english'))

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


## Part 1
def sortTuple(tup):    
    return(sorted(tup, key = lambda x: x[1], reverse=True))

wordCountFake = []
for word in set(fake_words):
	# if word not in stopWords:
	wordCountFake.append((word, fake_words.count(word)))
		
		# if word == 'hillary' or word == 'trump' or word == 'clinton' or word == 'donald':
		# 	print(word, fake_words.count(word))

wordCountReal = []
for word in set(real_words):
	# if word not in stopWords:
	wordCountReal.append((word, real_words.count(word)))

		# if word == 'hillary' or word == 'trump' or word == 'clinton' or word == 'donald':
		# 	print(word, real_words.count(word))

fake10 = sortTuple(wordCountFake)[0:10]
real10 = sortTuple(wordCountReal)[0:10]

words_fake = [x[0] for x in fake10]
nums_fake = [x[1] / len(fake_words) for x in fake10]

words_real = [x[0] for x in real10]
nums_real = [x[1] / len(real_words) for x in real10]

# proportion for top 10 words
plt.subplot(1, 2, 1)
plt.bar(words_fake, height = nums_fake)
plt.ylabel('Word Proportion')
plt.title('Top 10 Fake Words')
plt.subplot(1, 2, 2)
plt.bar(words_real, height = nums_real)
plt.title('Top 10 Real Words')
plt.xlabel('')
# plt.show()

# num of different words total
numFakeWords = len(set(fake_words))
numRealWords = len(set(real_words))
# print(numFakeWords, numRealWords)

## Part 2
headlines = fake.split('\n')
fakeLength = len(headlines)
tag = ['fake'] * len(headlines)
headlines.extend(real.split('\n'))
tag.extend(['real'] * (len(headlines) - fakeLength))

headlines = np.array(headlines)
tag = np.array(tag)


trainProp = 0.75
valProp = 0.15

trainNum = int(len(headlines) * trainProp)
valNum = int(len(headlines) * valProp)

idx = np.random.RandomState(seed=31).permutation(range(len(headlines)))
trainIdx = idx[:trainNum]
testIdx = idx[trainNum:valNum]
validIdx = idx[valNum:len(headlines)]


train_x = headlines[trainIdx]
train_y = tag[trainIdx]

test_x = headlines[testIdx]
test_y = tag[testIdx]

valid_x = headlines[validIdx]
valid_y = tag[validIdx]

def makeDict(headlines, tags, fake_mark):
	dictionary = {}
	for i in range(len(tags)):
		if tags[i] == fake_mark:
				words = headlines[i].split(' ')
				for word in set(words):
					if word in dictionary.keys():
						dictionary[word] += 1
					else:
						dictionary[word] = 1
	return dictionary
# print(makeDict(train_x, train_y, 'fake'))

train_fake_dict = makeDict(train_x, train_y, 'fake')
train_real_dict = makeDict(train_x, train_y, 'real')

count_c_fake = len(train_y[train_y == 'fake'])
count_c_real = len(train_y[train_y == 'real'])


#new dictionary with probabilites
def getProbs(m, pHat, headlines, tags, fake_mark):
	dictionary = makeDict(headlines, tags, fake_mark)
 	dict_probs = {}
 	for word in dictionary.keys():
		dict_probs[word] = (dictionary[word] + m*pHat)/(len(headlines[tags == fake_mark]) + m)
	return dict_probs
# dict_probs = {}
# for word in dictionary.keys():
# 	dict_probs[word] = (dictionary[word] + m*pHat)/(len(train_y[train_y == 'fake']) + m)
# if word in dictionary.keys():
# 	count_xc = dictionary[word]
# else:
# 	count_xc = 0
# count_xc = dictFake[word]

	# return (probfake)



m = range(0, 11, 1)
pHat = np.arange(0, 0.55, 0.05)

# for i in m:
# 	for j in pHat:
# 		print(i, j)

#which has highest accuracy rate on validation set
