# In this example we are going to create a simple HTML
# page with 2 input fields (numbers), and a link.
# Using jQuery we are going to send the content of both
# fields to a route on our application, which will
# sum up both numbers and return the result.
# Again using jQuery we'l show the result on the page


# We'll render HTML templates and access data sent by GET
# using the request object from flask. jsonigy is required
# to send JSON as a response of a request
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn import neighbors, datasets
import sys, thread, time, random

def CenterData(X):
    allXCoordinates = X[0,::3]
    meanValue = allXCoordinates.mean()
    X[0,::3] = allXCoordinates - meanValue

    allYCoordinates = X[0,1::3]
    meanValue = allYCoordinates.mean()
    X[0,1::3] = allYCoordinates - meanValue

    allZCoordinates = X[0,2::3]
    meanValue = allZCoordinates.mean()
    X[0,2::3] = allZCoordinates - meanValue
    return X

# Initialize the Flask application
app = Flask(__name__)


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/_knn')
def knn():
    a = request.args.get('a', 'hi', type=str)
    clf = pickle.load( open("userData/classifier.p", "rb") ) # Load data from saved classifier
    testData = np.zeros((1,30),dtype='f')

    a = a.split(',')

    for i in range(0,30):
        testData[0,i] = float(a[i]);

    testData = CenterData(testData)

    predictedClass = clf.predict( testData ) # Finds the predicted sign language number    

    num = int(predictedClass[0])

    return jsonify(result= num)

@app.route('/testendpoint')
def testendpoint():
	# 12 is the default value if none is there
	num = request.args.get("num", 12, type=int)
	return jsonify(result=num+1)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("80"),
        debug=True
    )