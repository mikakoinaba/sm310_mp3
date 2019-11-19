import nltk
# from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from math import exp, log

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
valNum = trainNum + int(len(headlines) * valProp)

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

# dictionary of counts per word (how many headlines it appears it)
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

# dictionary of probabilites per word given fake_mark
def getProbs(m, pHat, headlines, tags, fake_mark, trainDict):
	dict_probs = {}
	for word in trainDict.keys():
		dict_probs[word] = (trainDict[word] + m*pHat)/(len(headlines[tags == fake_mark]) + m)
	return dict_probs


def getPrediction(words, fakeProbs, realProbs, fakeZero, realZero):
	logSumFake = 0

	for word in words:
		val = 0
		if word not in fakeProbs:
			val = fakeZero
		else:
			val = fakeProbs[word]
		logSumFake += log(val)
	fakeProb = exp(logSumFake) * len(train_y[train_y == 'fake'])

	logSumReal = 0
	for word in words:
		val = 0
		if word not in realProbs:
			val = realZero
		else:
			val = realProbs[word]
		logSumReal += log(val)
	realProb = exp(logSumReal) * len(train_y[train_y == 'real'])

	if fakeProb >= realProb:
		return 'fake'
	else:
		return 'real'


m = range(1, 11, 1)
pHat = np.arange(0.05, 0.55, 0.05)


trainFake = makeDict(train_x, train_y, 'fake')
trainReal = makeDict(train_x, train_y, 'real')
# accuracy = []
# mpPairs = []
# for i in m:
# 	for j in pHat:
# 		mpPairs.append((i,j))
# 		pred = []
# 		fakeProbs = getProbs(i, j, train_x, train_y, 'fake', trainFake)
# 		fakeZero = (i*j)/(len(train_y[train_y == 'fake']) + i)
# 		realProbs = getProbs(i, j, train_x, train_y, 'real', trainReal)
# 		realZero = (i*j)/(len(train_y[train_y == 'real']) + i)

# 		for headline in valid_x:
# 			words = headline.split(' ')
# 			pred.append(getPrediction(words, fakeProbs, realProbs, fakeZero, realZero))

# 		# get accuracy of prediction and add to accuracy list
# 		same = 0
# 		for i in range(len(pred)):
# 			same += int(pred[i] == valid_y[i])
# 		accuracy.append(same / len(valid_y))


# maxAccuracyIndex = accuracy.index(max(accuracy))
# print(mpPairs[maxAccuracyIndex])

# m = 1
# pHat = 0.05


# given real, top 10 words
probReal = getProbs(1, 0.05, train_x, train_y, 'real', trainReal)
realStrong10 = sorted(probReal, key=probReal.get, reverse=True)[:10]
probFake = getProbs(1, 0.05, train_x, train_y, 'fake', trainFake)
fakeStrong10 = sorted(probFake, key=probFake.get, reverse=True)[:10]

# Part 1, 2, 3
# Part 4: conditional independence?
# Part 5

print(realStrong10)
print(fakeStrong10)




pred = []
fakeProbs = getProbs(1, 0.05, train_x, train_y, 'fake', trainFake)
fakeZero = (0.05)/(len(train_y[train_y == 'fake']) + 1)
realProbs = getProbs(1, 0.05, train_x, train_y, 'real', trainReal)
realZero = (0.05)/(len(train_y[train_y == 'real']) + 1)

for headline in test_x:
	words = headline.split(' ')
	pred.append(getPrediction(words, fakeProbs, realProbs, fakeZero, realZero))


# get accuracy of prediction and add to accuracy list
same = 0
for i in range(len(pred)):
	same += int(pred[i] == test_y[i])
accuracy = same / len(test_y)

print(len(test_y))
print(accuracy)

