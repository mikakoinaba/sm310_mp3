import nltk
# from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from math import exp, log
from random import random
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable
import torch

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

trainProp = 0.70
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
		val = 1 - ((fakeDict[word] + m*pHat) / (fakeCount + m))
		# if X_word = 1
		if word in words:
			uniqueFake.remove(word)
			val = 1 - val
		logSumFake += log(val)

	# for new word not seen before
	for i in range(len(uniqueFake)):
		logSumFake += log( m*pHat / (fakeCount + m))

	logSumFake += log(probFake)
	fakeProb = exp(logSumFake)

	logSumReal = 0
	uniqueReal = set(words)

	# for each of the 1 - N words, calculate probability
	for word in realDict:
		val = 1 - ((realDict[word] + m*pHat) / (realCount + m))
		if word in words:
			uniqueReal.remove(word)
			val = 1 - val
		logSumReal += log(val)

	for i in range(len(uniqueReal)):
		logSumReal += log(m*pHat / (realCount + m))

	logSumReal += log((1 - probFake))
	realProb = exp(logSumReal)

	# if the P(tag = 'fake' | headline) >= P(tag = 'real' | headline), predict 'fake'
#	print('fake ', fakeProb, 'real ', realProb)
	if fakeProb >= realProb:
		return 'fake'
	else:
		return 'real'

# return maxAccuracy and associated m and pHat
def getAccuracy(m, pHat, fakeDict, fakeCount, realDict, realCount, probFake, setHeadlines, setTags):
	accuracy = []
	mpPairs = []
	for i in m:
		for j in pHat:
			mpPairs.append((i,j))
			pred = []

			# for each headline in validation, add prediction 
			for headline in setHeadlines:
				words = headline.split(' ')
				pred.append(getPred(words, i, j, fakeDict, fakeCount, realDict, realCount, probFake))

			# for each m pHat pair, add accuracy
			same = 0
			for i in range(len(pred)):
				same += int(pred[i] == setTags[i])
			accuracy.append(same / len(setTags))

	maxAccuracy = max(accuracy) # find the max accuracy
	maxAccuracyIndex = accuracy.index(maxAccuracy) # get index of max accuracy
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

m = range(1, 20, 1) # want to avoid m = 0 
pHat = np.arange(0.01, 0.5, 0.02) # want to avoid pHat = 0

## just a test on the first two headlines
# words = valid_x[0].split(' ')
# print(getPred(words, 7, 0.01, fakeCounts, countFake, realCounts, countReal, probFake))

# words = valid_x[1].split(' ')
# print(getPred(words, 7, 0.01, fakeCounts, countFake, realCounts, countReal, probFake))

## run to do all the m pHat combinations
# print(getAccuracy(m, pHat, fakeCounts, countFake, realCounts, countReal, probFake, valid_x, valid_y))

mOpt = 19
pHatOpt = 0.01

#print(getAccuracy([mOpt], [pHatOpt], fakeCounts, countFake, realCounts, countReal, probFake, train_x, train_y))
#print(getAccuracy([mOpt], [pHatOpt], fakeCounts, countFake, realCounts, countReal, probFake, test_x, test_y))
# Report the performance on the training and the test sets using the parameters  from vlaidation 

# given real, top 10 words
# P('real' | X_word = 1)
# words that maximize P(X_word = 1 | real) * P(real) / P(X_word = 1 | fake) * P(fake) 
def presenceGivenWordReal (fakeDict, realDict, m, pHat, realCount, fakeCount, probReal):
	probs = {}

	for word in realDict:
		prob = ((realDict[word] + m*pHat) / (realCount + m)) * probReal

		if word in fakeDict:
			prob = prob / (((fakeDict[word] + m*pHat) / (fakeCount + m)) * (1-probReal))
		else:
			prob = prob / ((m*pHat / (fakeCount + m)) * (1-probReal))
		probs[word] = prob
	return probs

# words that maximize P(X_word = 1 | fake) * P(fake) / P(X_word = 1 | real) * P(real) 
def presenceGivenWordFake (fakeDict, realDict, m, pHat, realCount, fakeCount, probFake):
	probs = {}
	for word in fakeDict:
		prob = ((fakeDict[word] + m*pHat) / (fakeCount + m)) * probFake

		if word in realDict:
			prob = prob / (((realDict[word] + m*pHat) / (realCount + m)) * (1-probFake))
		else:
			prob = prob / ((m*pHat / (realCount + m)) * (1 - probFake))
		probs[word] = prob
	return probs

# words that maximize P(X_word = 0 | real) * P(real) / P(X_word = 0 | real) * P(real) 
def absentGivenWordReal (fakeDict, realDict, m, pHat, realCount, fakeCount, probReal):
	probs = {}

	for word in realDict:
		prob = (realDict[word] + m*pHat) / (realCount + m)
		prob = (1 - prob) * probReal

		if word in fakeDict:
			tempProb = 1 - ((fakeDict[word] + m*pHat) / (fakeCount + m))
			prob = prob / ( tempProb * (1-probReal))
		else:
			tempProb = 1 - ( m*pHat / (fakeCount + m))
			prob = prob / ( tempProb * (1-probReal))
		probs[word] = prob
	return probs

# words that maximize P(X_word = 0 | fake) * P(fake) / P(X_word = 0 | real) * P(real) 
def absentGivenWordFake (fakeDict, realDict, m, pHat, realCount, fakeCount, probFake):
	probs = {}
	for word in fakeDict:
		prob = (fakeDict[word] + m*pHat) / (fakeCount + m)
		prob = (1 - prob) * probFake

		if word in realDict:
			tempProb = 1 - ((realDict[word] + m*pHat) / (realCount + m))
			prob = prob / ( tempProb * (1 - probFake))
		else:
			tempProb = 1 - ( m*pHat / (realCount + m))
			prob = prob / (tempProb * (1 - probFake))
		probs[word] = prob
	return probs

# presence -> real
presenceReal = presenceGivenWordReal(fakeCounts, realCounts, mOpt, pHatOpt, countReal, countFake, probReal)
realStrong10P = sorted(presenceReal, key=presenceReal.get, reverse=True)[:10]

# presence -> fake
presenceFake = presenceGivenWordFake(fakeCounts, realCounts, mOpt, pHatOpt, countReal, countFake, probFake)
fakeStrong10P = sorted(presenceFake, key=presenceFake.get, reverse=True)[:10]

# print('presence -> real', realStrong10P)
# for word in realStrong10P:
# 	print(word, presenceReal[word])

# print('presence -> fake', fakeStrong10P)
# for word in fakeStrong10P:
# 	print(word, presenceFake[word])

# absence -> real
absenceReal = absentGivenWordReal(fakeCounts, realCounts, mOpt, pHatOpt, countReal, countFake, probReal)
realStrong10A = sorted(absenceReal, key=absenceReal.get, reverse=True)[:10]

# absence -> fake
absenceFake = absentGivenWordFake(fakeCounts, realCounts, mOpt, pHatOpt, countReal, countFake, probFake)
fakeStrong10A = sorted(absenceFake, key=absenceFake.get, reverse=True)[:10]

# print('absence -> real', realStrong10A)
# for word in realStrong10A:
# 	print(word, absenceReal[word])

# print('absence -> fake', fakeStrong10A)
# for word in fakeStrong10A:
# 	print(word, absenceFake[word])

# Part 5:
# lengthFake = 0
# lengthReal = 0
# for i in range(len(train_x)):
# 	if train_y[i] == 'real':
# 		lengthReal += len(train_x[i].split(' '))
# 	else:
# 		lengthFake += len(train_x[i].split(' '))

# avgFake = int(lengthFake / countFake) # 12
# avgReal = int(lengthReal / countReal) # 8
# print(avgReal, avgFake)

def probLists (m, pHat, countDict, count):
	wordList = []
	probList = []
	for word in countDict:
		wordList.append(word)
		probList.append((countDict[word] + m*pHat) / (count + m))
	return (wordList, probList)

fakeProbLists = probLists(mOpt, pHatOpt, fakeCounts, countFake)
realProbLists = probLists(mOpt, pHatOpt, realCounts, countReal)

# total = 0
# for prob in fakeProbLists[1]:
# 	total += prob
# 	print(prob)
# print('TOTAL: ',total)

def genHeadline(lists):
	headline = ''
	for i in range(len(lists[0])):
		if random() < lists[1][i]:
			headline += (' ' + lists[0][i])
#		word = np.random.choice(lists[0], p=lists[1]/sum(lists[1]))
	return headline[1:]

# print(genHeadline(fakeProbLists))
# print(genHeadline(realProbLists))

# Part 6
# at least 10...

#########################################################################################################################

# Part 7
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

# print(len(headlines))
vectorizer = CountVectorizer()
words = vectorizer.fit_transform(headlines).toarray()
wordSet = vectorizer.get_feature_names()
# print(words)

# 0 == fake, 1 == real
tags = []
for label in tag:
	tags.append(int(label == 'real'))
tags = np.array(tags)
# tags = vectorizer.fit_transform(tag).toarray()
# print(tags)

train_xLR = words[trainIdx]
train_yLR = tags[trainIdx]

test_xLR = words[testIdx]
test_yLR = tags[testIdx]

valid_xLR = words[validIdx]
valid_yLR = tags[validIdx]

# print(train_xLR)
# print(train_yLR)

x_train = Variable(torch.from_numpy(train_xLR), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(train_yLR), requires_grad=False).type(dtype_long)

x_valid = Variable(torch.from_numpy(valid_xLR), requires_grad=False).type(dtype_float)
y_validclasses = Variable(torch.from_numpy(valid_yLR), requires_grad=False).type(dtype_long)


x_test = Variable(torch.from_numpy(test_xLR), requires_grad=False).type(dtype_float)

dim_x = words.shape[1]
dim_out = 2

model_logreg = torch.nn.Sequential(
	torch.nn.Linear(dim_x, dim_out)
)

loss_train = []
loss_valid = [] 

learning_rate = 1e-3
N = 10000 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)

for t in range(N):
    y_pred = model_logreg(x_train)
    loss = loss_fn(y_pred, y_classes)
    loss_train.append(loss.data.numpy().reshape((1,))[0]/len(y_classes))

    y_validpred = model_logreg(x_valid)
    loss_v = loss_fn(y_validpred, y_validclasses)
    loss_valid.append(loss_v.data.numpy().reshape((1,))[0]/len(y_validclasses))

    model_logreg.zero_grad() 
    loss.backward()   
    optimizer.step()   

plt.plot(range(N), loss_valid, color='b', label='validation')
plt.plot(range(N), loss_train, color='r', label='training')
plt.legend(loc='upper left')
plt.title("Learning curve for training and validation sets")
plt.xlabel("Number of iterations (N)")
plt.ylabel("Cross entropy loss")
# plt.show()
# something's wrong... ^

# y_testpred = model_logreg(x_test).data.numpy()

# results on test set
# print('accuracy on testing set: ', np.mean(np.argmax(y_testpred, 1) == test_yLR))

# weights (wn) from word_i to 'fake' (z0)
weight_w = model_logreg[0].weight[0,:]

# weights (vn) from word_i to 'real' (z1)
weight_v = model_logreg[0].weight[1,:]

# Part 8
# changeDict = {}

# for i in len(wordSet):
# 	changeDict[wordSet[i]] = exp(weight_w[i]) / exp(weight_w[i])+ exp(weight_v[i])

# top10 = sorted(changeDict, key=changeDict.get, reverse=True)
# print(top10)


# find top 10 changes
# the other cases are basically the same code

# word weight from 0 -> 1 or 1 -> 0
# z0 = 0
# z1 = 0

# word0 = exp(z0) / (exp(z0) + exp(z1))
# word1 = exp(z0 + w1) / (exp(z0 + w1) + exp(z1 + v1))
# top 10 of word1 / word0


# top 10 presence predict real
# for each word exp(w1) / exp(w1)+ exp(v1)

# top 10 absence predict real
# for each word exp(-w1) / exp(-w1) + exp(-v1)

# top 10 presence predict fake
# for each word exp(v1) / exp(w1) + exp(v1)

# top 10 absence predict fake
# for each word exp(-v1) / exp(-w1) + exp(-v1)