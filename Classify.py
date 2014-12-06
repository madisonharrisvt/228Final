import numpy as np
import pickle
import matplotlib
import Leap, sys, thread, time, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors, datasets

#gestureData = pickle.load( open("userData/gesture.p", "rb"))
train0 = pickle.load( open("userData/Bongard_train0.p", "rb") )
test0 = pickle.load( open("userData/Bongard_test0.p", "rb") )
train1 = pickle.load( open("userData/Bongard_train1.p", "rb") )
test1 = pickle.load( open("userData/Bongard_test1.p", "rb") )
train6 = pickle.load( open("userData/train6.p", "rb") )
train4 = pickle.load( open("userData/train4.p", "rb") )
test6 = pickle.load( open("userData/test6.p", "rb") )
test4 = pickle.load( open("userData/test4.p", "rb") )
test5 = pickle.load( open("userData/Strayer_test5.p", "rb") )
train5 = pickle.load( open("userData/Strayer_train5.p", "rb") )
train2 = pickle.load( open("userData/Bongard_train2.p", "rb") )
test2 = pickle.load( open("userData/Bongard_test2.p", "rb") )
train3 = pickle.load( open("userData/Bishop_train3.p", "rb") )
test3 = pickle.load( open("userData/Bishop_test3.p", "rb") )
train7 = pickle.load( open("userData/Bongard_train7.p", "rb") )
test7 = pickle.load( open("userData/Bongard_test7.p", "rb") )
train8 = pickle.load( open("userData/Bongard_train8.p", "rb") )
test8 = pickle.load( open("userData/Bongard_test8.p", "rb") )
train9 = pickle.load( open("userData/Garcia_train9.p", "rb") )
test9 = pickle.load( open("userData/Garcia_test9.p", "rb") )

def ReshapeData(set1,set2,set3,set4,set5,set6,set7,set8,set9,set10):
	X = np.zeros((10000,5*2*3),dtype='f')
	y = np.zeros((10000),dtype='f')

	for i in range(0,1000):
		n = 0
		for j in range(0,5):
			for k in range(0,2):
				for m in range(0,3):
					X[i,n] = set1[j,k,m,i]
					y[i] = 4
					y[i+1000] = 6
					X[i+1000,n] = set2[j,k,m,i]
					X[i+2000,n] = set3[j,k,m,i]
					y[i+2000] = 5
					X[i+3000,n] = set4[j,k,m,i]
					y[i+3000] = 0
					X[i+4000,n] = set5[j,k,m,i]
					y[i+4000] = 1
					X[i+5000,n] = set6[j,k,m,i]
					y[i+5000] = 2
					X[i+6000,n] = set7[j,k,m,i]
					y[i+6000] = 3
					X[i+7000,n] = set8[j,k,m,i]
					y[i+7000] = 7
					X[i+8000,n] = set9[j,k,m,i]
					y[i+8000] = 8
					X[i+9000,n] = set10[j,k,m,i]
					y[i+9000] = 9

					n = n+1
	return X,y

def ReduceData(X):
	X = np.delete(X,1,1)
	X = np.delete(X,1,1)
	X = np.delete(X,0,2)
	X = np.delete(X,0,2)
	X = np.delete(X,0,2)

	return X

def CenterData(X):
	allXCoordinates = X[:,:,0,:]
	meanValue = allXCoordinates.mean()
	X[:,:,0,:] = allXCoordinates - meanValue

	allYCoordinates = X[:,:,1,:]
	meanValue = allYCoordinates.mean()
	X[:,:,1,:] = allYCoordinates - meanValue

	allZCoordinates = X[:,:,2,:]
	meanValue = allZCoordinates.mean()
	X[:,:,2,:] = allZCoordinates - meanValue
	return X	

train6 = ReduceData(train6)
train4 = ReduceData(train4)
test6 = ReduceData(test6)
test4 = ReduceData(test4)
train5 = ReduceData(train5)
test5 = ReduceData(test5)
train0 = ReduceData(train0)
test0 = ReduceData(test0)
train1 = ReduceData(train1)
test1 = ReduceData(test1)
train2 = ReduceData(train2)
test2 = ReduceData(test2)
train3 = ReduceData(train3)
test3 = ReduceData(test3)
train7 = ReduceData(train7)
test7 = ReduceData(test7)
train8 = ReduceData(train8)
test8 = ReduceData(test8)
train9 = ReduceData(train9)
test9 = ReduceData(test9)


train6 = CenterData(train6)
train4 = CenterData(train4)
test6 = CenterData(test6)
test4 = CenterData(test4)
train5 = CenterData(train5)
test5 = CenterData(test5)
train0 = CenterData(train0)
test0 = CenterData(test0)
train1 = CenterData(train1)
test1 = CenterData(test1)
train2 = CenterData(train2)
test2 = CenterData(test2)
train3 = CenterData(train3)
test3 = CenterData(test3)
train7 = CenterData(train7)
test7 = CenterData(test7)
train8 = CenterData(train8)
test8 = CenterData(test8)
train9 = CenterData(train9)
test9 = CenterData(test9)


trainX,trainy= ReshapeData(train4,train6,train5,train0,train1,train2,train3,train7,train8,train9)
testX,testy= ReshapeData(test4,test6,test5,test0,test1,test2,test3,test7,test8,test9)

clf = neighbors.KNeighborsClassifier(15)
clf.fit(trainX,trainy)

actualClass = trainy[0]
prediction = clf.predict(trainX[0,:])
numCorrect = 0

for i in range(10000):
	prediction = int( clf.predict( testX[i,:] ) )
	actualClass = int(testy[i])

	if(prediction == actualClass):
		numCorrect += 1

percentage = (float(numCorrect)/float(10000))*100.00

print percentage

pickle.dump(clf, open("userData/classifier.p","wb"))
