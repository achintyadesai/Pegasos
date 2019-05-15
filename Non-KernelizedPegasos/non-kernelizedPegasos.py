from array import *
import random
import numpy
def get_data():
	all_data = []
	train = open('fashion-mnist_train.csv', 'r')
	for index, data in enumerate(train):
	    all_data.append(data)
	print ("There are " + str(len(all_data)) + " words in all the Data Set")
	return all_data

def get_test_data():
	test_data = []
	test = open('fashion-mnist_test.csv', 'r')
	for index, data in enumerate(test):
	    test_data.append(data)
	print ("There are " + str(len(test_data)) + " words in all the Data Set")
	return test_data

def format_data(all_data):
	print (len(all_data))
	label = []
	xvector = []
	numberofentries = 0
	for i in range(len(all_data)):
		if all_data[i][0] == '0' or all_data[i][0] == '1':
			numberofentries = numberofentries + 1
			labelvalue = ord(all_data[i][0])-48
			if labelvalue == 0:
				labelvalue = -1
			label.append(labelvalue)

			firstcommaindex = all_data[i].find(",")
			if firstcommaindex == -1:
				print("Dataset error! No comma found in parsing")
			else:
				intermediateline = all_data[i][firstcommaindex+1:]
				xvector.append([int(x) for x in intermediateline.split(',')])
	print ("Number of entries:",numberofentries)
	return [label, xvector]

def multiplybyscalar(vectorlist, scalar) : 
	for i in range(len(vectorlist)):
		vectorlist[i] = vectorlist[i] * scalar
	return vectorlist 

def addscalar(vectorlist, scalar) : 
	for i in range(len(vectorlist)):
		vectorlist[i] = vectorlist[i] + scalar
	return vectorlist 

def addvector(vector1, vector2):
	vector3 = []
	for i in range(len(vector1)):
		vector3.append(vector1[i]+vector2[i])
	return vector3

def nonkernalized_pegasos(sampledataset,lambdavalue,totaliterations):
	labels = sampledataset[0]
	xvector = sampledataset[1]
	w = numpy.zeros(len(xvector[0]))
	totalsamples = len(labels)
	for iterationnumber in range(totaliterations):
		randomchoice = random.randint(0,totalsamples-1)
		learningrate = 1/(lambdavalue*(iterationnumber+1))
		print ("Before Dot Product:", numpy.shape(w), numpy.shape(xvector[randomchoice]))
		dotproduct = w.dot(numpy.asarray(xvector[randomchoice]))
		print ("DotProduct",dotproduct)
		yvalue = labels[randomchoice]
		if yvalue*dotproduct < 1:
			w = numpy.add(multiplybyscalar(w,(1-(learningrate*lambdavalue))),(multiplybyscalar(xvector[randomchoice],(learningrate*yvalue))))
		else:
			w = multiplybyscalar(w,(1-(learningrate*lambdavalue)))
		print("Wvalue",w)
		hingeloss = 1-yvalue*dotproduct
		print("HingeLoss", hingeloss)
	return w

def maxvalue(x,y):
	if x>y:
		return x
	else:
		return y

def pegasos_testing(w,testdataset):
	actuallabels = testdataset[0]
	xvector = testdataset[1]
	totaliterations = len(testdataset[0])
	correctpredictions = 0
	wrongpredictions = 0
	for iterationnumber in range(totaliterations):
		dotproduct = w.dot(numpy.asarray(xvector[iterationnumber]))
		print ("Predicted value", dotproduct, "Actual value", actuallabels[iterationnumber])
		if dotproduct<0 and actuallabels[iterationnumber]==-1:
			correctpredictions = correctpredictions + 1
		elif dotproduct>0 and actuallabels[iterationnumber]==1:
			correctpredictions = correctpredictions + 1
		else :
			wrongpredictions = wrongpredictions + 1
	accuracy = correctpredictions/totaliterations
	return [accuracy, wrongpredictions/totaliterations]


datapoints = format_data(get_data())
print ("Shape",numpy.shape(datapoints))
print ("Length of labels", len(datapoints[0]))
print ("Number of xvalues", len(datapoints))
print ("length of an xvalue", len(datapoints[1][0]))
parameter1 = 0.125
totaliteration = 1000
batchsize = 50

w = nonkernalized_pegasos(datapoints,parameter1,totaliteration)

accuracy,wrongpredictratio = pegasos_testing(w,format_data(get_test_data()))

print ("Final w value", w)

print ("Accuracy", accuracy)

print ("Wrong Prediction ratio", wrongpredictratio)

