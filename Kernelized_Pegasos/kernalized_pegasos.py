from array import *
import random
import numpy

def get_data():
	#global training_data, validation_data, test_data # Editing the global varabile
	#print "		-> Getting Test, Validation & Training Data from {DATA SOURCE}"
	#startTime = time.time() # Starts Timer
	all_data = []
	#training_data = []
	#test_data = []
	train = open('fashion-mnist_train.csv', 'r')
	#test = open('fashion-mnist_test.csv', 'r')
	for index, data in enumerate(train): # Getting data from the files
	    all_data.append(data)
	    #if index < 60000: # Getting 0-3999 data to training set
	    #training_data.append(data)
	    #else: # Getting 4000-5000 data to validation set
	    #validation_data.append(data)
	#for testdata in test: # Getting test data from the files
	#if testdata: # Getting 0-1000 data to test set
	    #test_data.append(testdata) # put data into test_data array
	#print "			-> Took " + str(time.time() - startTime) + " seconds"
	print ("			-> There are " + str(len(all_data)) + " words in all the Data Set")
	#print ("			-> There are " + str(len(training_data)) + " words in the Training Data Set")
	#print "			-> There are " + str(len(validation_data)) + " words in the Validation Data Set"
	#print ("			-> There are " + str(len(test_data)) + " words in the Test Data Set")
	#print (all_data[1])
	#print ("OTHELLO")
	#print (all_data[2])
	return all_data

def get_test_data():
	#global training_data, validation_data, test_data # Editing the global varabile
	#print "		-> Getting Test, Validation & Training Data from {DATA SOURCE}"
	#startTime = time.time() # Starts Timer
	#all_data = []
	#training_data = []
	test_data = []
	#train = open('fashion-mnist_train.csv', 'r')
	test = open('fashion-mnist_test.csv', 'r')
	for index, data in enumerate(test): # Getting data from the files
	    test_data.append(data)
	    #if index < 60000: # Getting 0-3999 data to training set
	    #training_data.append(data)
	    #else: # Getting 4000-5000 data to validation set
	    #validation_data.append(data)
	#for testdata in test: # Getting test data from the files
	#if testdata: # Getting 0-1000 data to test set
	    #test_data.append(testdata) # put data into test_data array
	#print "			-> Took " + str(time.time() - startTime) + " seconds"
	print ("			-> There are " + str(len(test_data)) + " words in all the Data Set")
	#print ("			-> There are " + str(len(training_data)) + " words in the Training Data Set")
	#print "			-> There are " + str(len(validation_data)) + " words in the Validation Data Set"
	#print ("			-> There are " + str(len(test_data)) + " words in the Test Data Set")
	#print (all_data[1])
	#print ("OTHELLO")
	#print (all_data[2])
	return test_data

def format_data(all_data):
	print (len(all_data))
	label = []
	xvector = []
	numberofentries = 0
	for i in range(len(all_data)):
		if all_data[i][0] == '0' or all_data[i][0] == '1':
			numberofentries = numberofentries + 1
			#print(all_data[i])
			labelvalue = ord(all_data[i][0])-48
			if labelvalue == 0:
				labelvalue = -1
			label.append(labelvalue)

			firstcommaindex = all_data[i].find(",")
			#print (firstcommaindex)
			if firstcommaindex == -1:
				print("Dataset error! No comma found in parsing")
			else:
				intermediateline = all_data[i][firstcommaindex+1:]
				#print (intermediateline)
				xvector.append([int(x) for x in intermediateline.split(',')])
	print ("Number of entries:",numberofentries)
	return [label, xvector]

'''def get_new_data():
    all_data = []
    training_data = []
    test_data = []
    all_data = genfromtxt('fashion-mnist_train.csv', delimiter=',')
    print (all_data)
'''

def multiplybyscalar(vectorlist, scalar) : 
    # Multiply elements one by one 
	for i in range(len(vectorlist)):
		vectorlist[i] = vectorlist[i] * scalar
	return vectorlist 

def addscalar(vectorlist, scalar) : 
    # Multiply elements one by one 
	for i in range(len(vectorlist)):
		vectorlist[i] = vectorlist[i] + scalar
	return vectorlist 

def addvector(vector1, vector2):
	vector3 = []
	for i in range(len(vector1)):
		vector3.append(vector1[i]+vector2[i])
	return vector3

def gaussian_kernel(vector1, vector2, gamma):
	
	#Kernel(x,y) = e^(-gamma*||x-y||^2_2)
	differencevector = numpy.array(vector1) - numpy.array(vector2)
	gamma = 20
	#print (differencevector)
	#print ("Gamma", gamma)
	exponent = gamma*differencevector.dot(differencevector)
	kernelvalue = 2.718281**exponent
	print ("Kernel", kernelvalue)
	return kernelvalue

def polynomial_kernel(vector1, vector2, degree):
	dotproduct = (numpy.array(vector1)).dot(numpy.array(vector2))
	
	#print(numpy.power(dotproduct, degree))
	return numpy.power(dotproduct, degree)

'''
def batchPegasos(dataSet, labels, lam, T, k):
	m, n = numpy.shape(dataSet);
	w = numpy.zeros(n);
	dataIndex = range(m)
	for t in range(1, T + 1):
		wDelta = mat(zeros(n))  # reset wDelta
		eta = 1.0 / (lam * t)
		random.shuffle(dataIndex)
		for j in range(k):  # go over training set
		    i = dataIndex[j]
		    p = predict(w, dataSet[i, :])  # mapper code
		    if labels[i] * p < 1:  # mapper code
		        wDelta += labels[i] * dataSet[i, :].A  # accumulate changes
		w = (1.0 - 1 / t) * w + (eta / k) * wDelta  # apply changes at each T
	return w
'''
def kernelized_pegasos(sampledataset,lambdavalue,totaliterations,gamma):
	labels = sampledataset[0]
	xvector = sampledataset[1]
	totalsamples = len(labels)
	alpha = [0]*totalsamples
	#print (totalsamples)
	#print (xvector[0])
	countincrements = 0
	for iterationnumber in range(totaliterations):
		print("Iteration ", iterationnumber, " Begins")
		#print(random.randint(0,totalsamples-1))
		randomchoice = random.randint(0,totalsamples-1)
		#randomchoice = 0
		yvalue = labels[randomchoice]
		coefficient = yvalue*(1/(lambdavalue*(iterationnumber+1)))
		sumoverj = 0
		for j in range(totalsamples):
			sumoverj = sumoverj + (alpha[j]*labels[j]*polynomial_kernel((xvector[j]),xvector[randomchoice],gamma))
		indicatorexpression = coefficient * sumoverj
		if indicatorexpression < 1:
			alpha[randomchoice] = alpha[randomchoice]+1
			countincrements = countincrements + 1
	print("Increments", countincrements)
	#print ("Alpha",alpha)
	return alpha

def kernelized_pegasos_testing(lambdavalue,totaliteration,alpha,datapoints,testpoints,gamma):
	learningrate = 1/(lambdavalue*totaliteration)
	labels = datapoints[0]
	xvector = datapoints[1]
	xtestpoints = testpoints[1]
	actuallabels = testpoints[0]
	w = []
	sum = 0
	correctpredictions = 0
	wrongpredictions = 0
	print ("Total testpoints", len(testpoints[0]))
#	for counter in range(len(testpoints[0])):
	for counter in range(100):
		print("Iteration ", counter, " Begins")
		for iterationnumber in range(len(alpha)):
			#print("Alpha", alpha[iterationnumber])
			#print("Label", labels[iterationnumber])
			#print("Kernel", polynomial_kernel(xvector[iterationnumber],xtestpoints[counter],gamma))
			sum = sum + ((alpha[iterationnumber]*labels[iterationnumber])*polynomial_kernel(xvector[iterationnumber],xtestpoints[counter],gamma))
		sum = sum * learningrate
		#print("Sum",sum)
		#print("Real", actuallabels[counter])
		if sum<0 and actuallabels[counter]==-1:
			correctpredictions = correctpredictions + 1
		elif sum>0 and actuallabels[counter]==1:
			correctpredictions = correctpredictions + 1
		else :
			wrongpredictions = wrongpredictions + 1
	accuracy = correctpredictions/100
	print ("Correctpredictions ", correctpredictions, "Wrongpredictions ", wrongpredictions)
	return [accuracy, wrongpredictions/100]

	

def nonkernalized_pegasos(sampledataset,lambdavalue,totaliterations):
	labels = sampledataset[0]
	xvector = sampledataset[1]
	w = numpy.zeros(len(xvector[0]))
	totalsamples = len(labels)
	#print (totalsamples)
	#print (xvector[0])
	for iterationnumber in range(totaliterations):
		#print(random.randint(0,totalsamples-1))
		randomchoice = random.randint(0,totalsamples-1)
		#randomchoice = 0
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


def nonkernalized_batched_pegasos(sampledataset,lambdavalue,totaliterations,batchsize):
	labels = sampledataset[0]
	xvector = sampledataset[1]
	w = numpy.zeros(len(xvector[0]))
	totalsamples = len(labels)
	#print (totalsamples)
	#print (xvector[0])
	for iterationnumber in range(totaliterations):
		#print(random.randint(0,totalsamples-1))
		hinge = numpy.zeros(len(xvector[0]))
		hingeloss = 0
		print("Iteration ", iterationnumber, " begins")
		for batchnumber in range(batchsize):
			randomchoice = random.randint(0,totalsamples-1)
			#randomchoice = iterationnumber+batchnumber
			dotproduct = w.dot(numpy.asarray(xvector[randomchoice]))
			#print ("DotProduct",dotproduct)
			yvalue = labels[randomchoice]
			if yvalue*dotproduct < 1:
				hinge = numpy.add(hinge, multiplybyscalar(xvector[randomchoice],yvalue))
				hingeloss = hingeloss + maxvalue(0, 1-(yvalue*dotproduct))

#			else:
#				batchnumber = batchnumber-1
#		sum = 0
		sum = numpy.sum(hinge)

		print("Loss", sum)
		#randomchoice = 0

		learningrate = 1/(lambdavalue*(iterationnumber+1))

		w = numpy.add(multiplybyscalar(w,(1-learningrate*lambdavalue)),(multiplybyscalar(hinge,(learningrate/batchsize))))

		#print("Wvalue",w)
		print("HingeLoss", hingeloss/batchsize)
	return w

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



#print("Hi")
#print("Hellothere")
datapoints = format_data(get_data())
print ("Shape",numpy.shape(datapoints))
print ("Length of labels", len(datapoints[0]))
print ("Number of xvalues", len(datapoints))
print ("length of an xvalue", len(datapoints[1][0]))
#print ("Labels", datapoints[0])
#print ("Xvector", datapoints[1][0])
#dataset = [[1,1],[3,1],[5,-1],[2,-1],[4,1]]
parameter1 = 1
totaliteration = 500
batchsize = 50
gamma = 1

#w = nonkernalized_pegasos(datapoints,parameter1,totaliteration)
alpha = kernelized_pegasos(datapoints,parameter1,totaliteration,gamma)
testpoints = format_data(get_test_data())
accuracy = kernelized_pegasos_testing(parameter1,totaliteration,alpha,datapoints,testpoints,gamma)
#accuracy = pegasos_testing(w,testpoints)

#print ("Final w value", w)
print ("Accuracy", accuracy[0], "Wrong Prediction ratio", accuracy[1])

#nonkernalized_batched_pegasos(datapoints,parameter1,totaliteration, batchsize)
#batchPegasos(datapoints[1],datapoints[0],parameter1,totaliteration, batchsize)
#S is a 2D array (x,y) of size [m,2]
