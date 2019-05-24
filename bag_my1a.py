import csv
import math
from random import seed
from random import randrange
from csv import reader
from operator import itemgetter
from random import randrange, seed
import numpy as np
from random import randrange, seed
import matplotlib.pyplot as plt

acc= []
test_data = []
x_test = []
y_test = []
acc1 = []

with open("test1.csv", encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    data = [dota for dota in rows]
    for x in range(len(data)):
        ins = []
        ins_y = []
        for y in range(4):
            ins.append(float(data[x][y]))

        c = data[x][4]
        x_test.append(ins)
        y_test.append(int(c))

def kfold(data, n):
    #print(len(data))
    #print(data)
    np.random.shuffle(data)
    #print(data)
    #k = len(data) / n
    k = 100
    #print(k)
    #print("data is ", data)

    samples = list()
    while (len(samples) < n):
        np.random.shuffle(data)
        #print("data", data)
        sample = list()
        #while len(sample) < k:
        for index in range(int(k)):
            index = randrange(len(data))
            #print("index is",index)
            #print("data index",data[index])
            sample.append(data[index])
        #print("sample is ", sample)
        #print(len(sample))
        samples.append(sample)
    #print("samples together ", samples)
    #print("length of samples", len(samples))
    return samples
def data_clean1(data):
    x_train = []
    y_train = []
    test_data = []
    for x in range(len(data)):
        ins = []
        ins_y = []
        for y in range(4):
            ins.append(float(data[x][y]))

        c = data[x][4]
        y_train.append(float(c))

        x_train.append(ins)


    #print(np.array(x_train))

    x = np.array(x_train)
    y = np.array(y_train)
    #print("x in data clean,", x)
    #print("y in data clean", y)
        # x_train =
        # y_train = y

    with open("test1.csv", encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        dota = [dota for dota in rows]

    for k in range(len(dota)):
        ins = []
        for l in range(4):
            ins.append(float(dota[k][l]))

        test_data.append(ins)
    test = np.array(test_data)
        # print(test_data)

    return x, y,test


def normalise_data(x):
    " we can normalise by subtracting the maximum value by the minimum value and  then divide by the standard deviation of x or divide by the range"

    max_x = np.max(x,axis = 0)
    min_x = np.min(x,axis = 0)
    normalised_x = 1 - (max_x - x)/(max_x - min_x)
    #print(normalised_x)
    return normalised_x
def calculate_sigmoid(x,weight):
    #print("calculate sigmoid x",x)
    #print("calculate sigmoid weight",weight)
    h = 1.0 / (1 + np.exp(-np.dot(x, weight.T)))
    return h

def dot_product(a,b):
    return sum(x*y for x,y in zip(a,b))

def calculate_mean(m):
    return (sum(x for x in m))/len(m)


def calculate_gradient(weight, x, y):
    # X = X - y.reshape(X.shape[0] -1)
    # print("y in gradient descent",y)
    # print("weight in gradient descent",weight)
    # print("x in gradient descent",x)
    h = calculate_sigmoid(x, weight)
    # print("h in gradient decent",h)
    first_calc = h - y.reshape(x.shape[0], -1)
    final_calc = np.dot(first_calc.T, x)
    # print("grad in gradient",final_calc)
    return final_calc


def calculate_cost(x, weight, y):
    h = calculate_sigmoid(x, weight)
    # print("y before squeeze",y)
    y = np.squeeze(y)

    h = np.mean((-(y * np.log(h)) - ((1 - y)) * np.log(1 - h)))

    return h


def grad_descent(x, y, weight, learning, gradient):
    cost = calculate_cost(x, weight, y)
    #print(cost)
    iteration, change = 1, 1
    while (change > gradient):
        # iteration += 1
        prev_cost = cost
        # finding the weights and updating it after every itertation. The loop goes on unti the condition is met
        weight = weight - (learning * calculate_gradient(weight, x, y))
        cost = calculate_cost(x, weight, y)
        # print("cost in grad",cost)

        change = prev_cost - cost
        # print("change in grad", change)
        iteration = iteration + 1
        # print("iteration no is",iteration)

        # h = calculate_sigmoid(x, weight)

        '''
        difference = h - y
        print("y in grad",y)
        gradient = calculate_gradient(difference, h, y)
        weight = minimise_loss(weight, learning, gradient)

        current_cost = calculate_cost(h, y)
        change = prev_cost - current_cost
        '''
    return weight, iteration

def predict_val(weight,x):
    prob = calculate_sigmoid(x,weight)
    prediction = np.where(prob >= .5, 1, 0)
    return np.squeeze(prediction)
def logistic_regression(data):


    X,y,test = data_clean1(data)
    X = normalise_data(X)
    one_column = np.ones((len(X), 1))
    #print("The normalised input data is", X)
    X = np.concatenate((one_column, X), axis=1)
    weight = np.matrix(np.zeros(5))
    # print("xin main", X)
    # print("y in main", y)
    # print("beta in main", weight)
    weight, n = grad_descent(X, y, weight, 0.01, 0.01)
    predict = predict_val(weight, X)
    #print(len(predict))
    #print("predict",predict)
    #print("the computed theta value is", weight)
    # print("difference between y and predicted y")
    #print("The acutal output for the given input data is where '1' represents male and '0' representd female", y)
    #print("The predicted output for the given data is", predict)
    #print("len of y",len(y))

    y_count = np.sum(y == predict)

    Accuracy = (y_count/len(y))*100
    #print("The calculated regression coefficients are", weight)
    #print("total no of iterations taken", n)
    #print("The count of the labels predicted", y_count)
    #print("Accuracy is",(y_count/len(y))*100)
    # print("y out",y)
    # fit(x,y,1)
    #acc.append(Accuracy)
    #plot_reg(X, y, weight)
    return X,y,weight,n

def bagging_algo():
    with open("train1.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]
        #print(len(data))
        data_splits = kfold(data, 5)
        model = []
        i = []
        p = 0
        for div_data in data_splits:
            #print("count is",p)
            p += 1
            X,y,weight,n = logistic_regression(div_data)
            model.append([X,y,weight,n])
        #print("MODEL LENGTH IS",len(model))
    return model


        #print(acc)
def test():
    test_data = []
    x_test = []
    y_test = []
    acc1 = []
    with open("test1.csv", encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        data = [dota for dota in rows]
        for x in range(len(data)):
            ins = []
            ins_y = []
            for y in range(4):
                ins.append(float(data[x][y]))

            c = data[x][4]
            x_test.append(ins)
            y_test.append(int(c))
    model = bagging_algo()
    X = x_test
    y = y_test
    X = normalise_data(X)
    one_column = np.ones((len(X), 1))
    # print("The normalised input data is", X)
    X = np.concatenate((one_column, X), axis=1)
    all_predictions = []
    for m in model:
        weight = m[2]
        predict = predict_val(weight, X)
        #print("predict is ",predict)
        all_predictions.append(predict)
        n = m[3]
        #print("The actual output for the given input data is where '1' represents male and '0' representd female", y)
        #print("The predicted output for the given data is", predict)
        #print("len of y", len(y))

        y_count = np.sum(y == predict)

        Accuracy = (y_count / len(y)) * 100
        #print("The calculated regression coefficients are", weight)
        #print("total no of iterations taken", n)
        #print("The count of the labels predicted", y_count)
        #print("Accuracy is", (y_count / len(y)) * 100)
        # print("y out",y)
                # fit(x,y,1)
        acc1.append(Accuracy)
    #print("length of model",len(model))
    print("Accuracy is",np.array(acc1))
    print("Avg of accuracy",np.average(np.array(acc1)))
    #print("all_predictions",all_predictions)
    temp = []
    avg_prediction = []

    '''

    for prediction in all_predictions:
        for i in prediction:
            temp.append(prediction[i])
        val = np.argmax(np.bincount(temp))
        
    print("value is ",val)
    '''
    myshape = str(np.array(all_predictions).shape)
    part1, part2 = myshape.split(', ')
    part1 = part1[1:]  # now is '50'
    part2 = part2[:-1]
    #print("part1",part1)
    #print("part2",part2)
    val = []
    for j in range(int(part2)):

        ins = []
        for i in range(int(part1)):

            ins.append(all_predictions[i][j])
        #print("ins",ins)
        val.append(np.argmax(np.bincount(ins)))
        temp.append(ins)

    avg_prediction.append(val)
    '''
    print("value is ",avg_prediction[0])
    print("actual value is ",y)
    '''
    y_count = 0
    for i in range(len(y)):
        if(y[i] == avg_prediction[0][i]):
            y_count += 1
    y_error = 0
    for i in range(len(y)):
        if (y[i] != avg_prediction[0][i]):
            y_error += 1


    #y_count = np.sum(y == avg_prediction[0])
    #print("y_count",y_count)

    Accuracy = (y_count / len(y)) * 100
    error = (y_error / len(y)) * 100

    print("The accuracy after bagging is ",Accuracy)
    print("The error rate for single classifier is",error)
    return avg_prediction


def calculate_avg_of_prediction(bagging_prediction):
    temp = []
    avg_prediction = []
    bagging_prediction = bagging_prediction
    myshape = str(np.array(bagging_prediction).shape)
    part1, part2 = myshape.split(', ')
    part1 = part1[1:]  # now is '50'
    part2 = part2[:-1]
    # print("part1",part1)
    # print("part2",part2)

    val = []
    for j in range(int(part2)):

        ins = []
        for i in range(int(part1)):
            ins.append(bagging_prediction[i][j])
        # print("ins",ins)
        val.append(np.argmax(np.bincount(ins)))
        temp.append(ins)

    avg_prediction.append(val)
    return avg_prediction


def calculate_accuracy(actual,predicted):

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[0][i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def get_error_rate(Y,pred):
    return sum(pred != Y) / float(len(Y))

test()














