import nltk
# from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from math import exp, log

def Part1():
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

#########################################################################################################################

## Part 2
fake = open('clean_fake.txt', encoding='utf-8').read()
real = open('clean_real.txt', encoding='utf-8').read()

headlines = fake.split('\n')[:-1] # last index is empty
fakeLength = len(headlines)
tag = ['fake'] * len(headlines)
headlines.extend(real.split('\n')[:-1]) # last index is empty
tag.extend(['real'] * (len(headlines) - fakeLength))

headlines = np.array(headlines)
tag = np.array(tag)

trainProp = 0.75
valProp = 0.15

trainNum = int(len(headlines) * trainProp) 
valNum = trainNum + int(len(headlines) * valProp)

idx = np.random.RandomState(seed=31).permutation(range(len(headlines)))
trainIdx = idx[:trainNum] # 0 - 2448 (trainNum - 1)
testIdx = idx[trainNum:valNum] # 2449 - 2937 (valNum - 1)
validIdx = idx[valNum:len(headlines)] # 2938 - 3265

train_x = headlines[trainIdx]
train_y = tag[trainIdx]

test_x = headlines[testIdx]
test_y = tag[testIdx]

valid_x = headlines[validIdx]
valid_y = tag[validIdx]

# dictionary of count(X_word = 1, tag = fake_mark)
# number of real / fake headlines a word appears in 
def countPerWord(headlines, tags, fake_mark):
	countDict = {}
	for i in range(len(headlines)):
		if tags[i] == fake_mark:
				words = headlines[i].split(' ')
				for word in set(words):
					if word in countDict:
						countDict[word] += 1
					else:
						countDict[word] = 1
	return countDict

# predict headline as real or fake
def getPred(words, m, pHat, fakeDict, fakeCount, realDict, realCount, probFake):
	logSumFake = 0
	uniqueFake = set(words)

	# for each of the 1 - N words, calculate probability
	for word in fakeDict:
		val = 1 - (fakeDict[word] + m*pHat) / (fakeCount + m)
		# if X_word = 1
		if word in words:
			uniqueFake.remove(word)
			val = 1 - val
		logSumFake += log(val)

	# for new word not in dict, predict pHat
	for i in range(len(uniqueFake)):
		logSumFake += log(pHat)

	logSumFake += log(probFake)
	fakeProb = exp(logSumFake)

	logSumReal = 0
	uniqueReal = set(words)

	# for each of the 1 - N words, calculate probability
	for word in realDict:
		val = 1 - (realDict[word] + m*pHat) / (realCount + m)
		if word in words:
			uniqueReal.remove(word)
			val = 1 - val
		logSumReal += log(val)

	for i in range(len(uniqueReal)):
		logSumReal += log(pHat)

	logSumReal += log((1 - probFake))
	realProb = exp(logSumReal)

	# if the P(tag = 'fake' | headline) >= P(tag = 'real' | headline), predict 'fake'
	print('fake ', fakeProb, 'real ', realProb)
	if fakeProb >= realProb:
		return 'fake'
	else:
		return 'real'

def getAccuracy(m, pHat, fakeCounts, countFake, realCounts, countReal, probFake):
	accuracy = []
	mpPairs = []
	for i in m:
		for j in pHat:
			mpPairs.append((i,j))
			pred = []

			# for each headline in validation, add prediction 
			for headline in valid_x:
				words = headline.split(' ')
				pred.append(getPred(words, i, j, fakeCounts, countFake, realCounts, countReal, probFake))

			# for each m pHat pair, add accuracy
			same = 0
			for i in range(len(pred)):
				same += int(pred[i] == valid_y[i])
			accuracy.append(same / len(valid_y))
	maxAccuracy = max(accuracy)
	maxAccuracyIndex = accuracy.index(maxAccuracy)
	return (maxAccuracy, mpPairs[maxAccuracyIndex])


# count(X_word = 1, tag = 'fake')
fakeCounts = countPerWord(train_x, train_y, 'fake')

# count(X_word = 1, tag = 'real')
realCounts = countPerWord(train_x, train_y, 'real')

# count(tag = 'fake')
countFake = sum(train_y == 'fake')

totalCount = len(train_y)
countReal = totalCount - countFake

probFake = countFake / totalCount
probReal = 1 - probFake

m = range(10)
pHat = np.arange(0.01, 0.5, 0.02)

## just a test on the first headline
words = valid_x[0].split(' ')
print(getPred(words, 7, 0.01, fakeCounts, countFake, realCounts, countReal, probFake))

## run to do all the m pHat combinations
# print(getAccuracy(m, pHat, fakeCounts, countFake, realCounts, countReal, probFake))

# # m = 7
# # pHat = 0.01


# # given real, top 10 words
# probReal = getProbs(1, 0.05, train_x, train_y, 'real', trainReal)
# realStrong10 = sorted(probReal, key=probReal.get, reverse=True)[:10]
# probFake = getProbs(1, 0.05, train_x, train_y, 'fake', trainFake)
# fakeStrong10 = sorted(probFake, key=probFake.get, reverse=True)[:10]

# # Part 1, 2, 3
# # Part 4: conditional independence?
# # Part 5

# print(realStrong10)
# print(fakeStrong10)



# dictinary of P(X_word = 1 and 0 | tag = fake_mark)
# def condProbPerWord(m, pHat, countDict, totalCount):
# 	exist = {}
# 	noExist = {}
# 	for word in countDict:
# 		exist[word] = (countDict[word] + m*pHat) / (totalCount + m)
# 		noExist[word] = (totalCount - countDict[word] + m*pHat) / (totalCount + m)
# 	return (exit, noExist)


# for when we find an optimal m and pHat
# fakeCond = condProbPerWord(0, 0, fakeCounts, countFake)
# realCond = condProbPerWord(0, 0, realCounts, countReal)

# # P(X_word = 1| tag = 'fake')
# yesFake = fakeCond[0]
# # P(X_word = 0| tag = 'fake')
# noFake = fakeCond[1]

# # P(X_word = 1| tag = 'real')
# yesReal = realCond[0]
# # P(X_word = 0| tag = 'real')
# noReal = realCond[1]


# pred = []
# fakeProbs = getProbs(1, 0.05, train_x, train_y, 'fake', trainFake)
# fakeZero = (0.05)/(len(train_y[train_y == 'fake']) + 1)
# realProbs = getProbs(1, 0.05, train_x, train_y, 'real', trainReal)
# realZero = (0.05)/(len(train_y[train_y == 'real']) + 1)

# for headline in test_x:
# 	words = headline.split(' ')
# 	pred.append(getPrediction(words, fakeProbs, realProbs, fakeZero, realZero))


# # get accuracy of prediction and add to accuracy list
# same = 0
# for i in range(len(pred)):
# 	same += int(pred[i] == test_y[i])
# accuracy = same / len(test_y)

# print(len(test_y))
# print(accuracy)

