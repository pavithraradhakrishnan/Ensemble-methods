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
from tqdm import tqdm

def data_clean1():
    x_train = []
    y_train = []
    test_data = []
    x_test = []
    y_test = []
    with open("train1.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]
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
    #print("x  in data clean", x)
    y = np.array(y_train)
    #print("x in data clean,", x)
    #print("y in data clean", y)
        # x_train =
        # y_train = y

    with open("test1.csv", encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        data = [dota for dota in rows]
        for a in range(len(data)):
            ins = []
            ins_y = []
            for b in range(4):
                ins.append(float(data[a][b]))

            c = data[a][4]
            x_test.append(ins)
            y_test.append(int(c))
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    return x, y,x_test,y_test



def normalise_data(x):
    " we can normalise by subtracting the maximum value by the minimum value and  then divide by the standard deviation of x or divide by the range"

    max_x = np.max(x, axis=0)
    min_x = np.min(x, axis=0)
    normalised_x = 1 - (max_x - x) / (max_x - min_x)
    # print(normalised_x)
    return normalised_x

def calculate_hypothesis_fn(x, weight):
    #print("calculate sigmoid x", x)
    #print("calculate sigmoid weight", weight)
    h = 1.0 / (1 + np.exp(-np.dot(x, weight.T)))
    return h

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def calculate_mean(self,m):
    return (sum(x for x in m)) / len(m)

def calculate_gradient(weight, x, y):
        # X = X - y.reshape(X.shape[0] -1)
        # print("y in gradient descent",y)
        # print("weight in gradient descent",weight)
        # print("x in gradient descent",x)
    h = calculate_hypothesis_fn(x, weight)
        # print("h in gradient decent",h)
    a = h - y.reshape(x.shape[0], -1)
    b = np.dot(a.T, x)
        # print("grad in gradient",final_calc)
    return b

def calculate_cost(x, weight, y):
    # calculating the hypothesis function
    h = calculate_hypothesis_fn(x, weight)
    # print("y before squeeze",y)
    y = np.squeeze(y)

    h = np.mean((-(y * np.log(h)) - ((1 - y)) * np.log(1 - h)))

    return h


def grad_descent(x, y, weight, lr, gradient):
    cost = calculate_cost(x, weight, y)
    # print(cost)
    num_iteration, diff = 1, 1
    while (diff > gradient):
        # iteration += 1
        prev_cost = cost
        # finding the weights and updating it after every itertation. The loop goes on unti the condition is met
        weight = weight - (lr * calculate_gradient(weight, x, y))
        cost = calculate_cost(x, weight, y)
        # print("cost in grad",cost)

        diff = prev_cost - cost
        # print("diff in grad", diff)
        num_iteration = num_iteration + 1
        # print("iteration no is",iteration)

        # h = calculate_sigmoid(x, weight)


    return weight, num_iteration

def predict_class(weight, x):
    prob = calculate_hypothesis_fn(x, weight)
    prediction = np.where(prob >= .5, 1, 0)
    return np.squeeze(prediction)

def logistic_regression(x_train,y_train):
    X, y = x_train,y_train
    #X = normalise_data(X)
    #one_column = np.ones((len(X), 1))
    # print("The normalised input data is", X)
    #X = np.concatenate((one_column, X), axis=1)
    weight = np.matrix(np.zeros(5))
    # print("xin main", X)
    # print("y in main", y)
    # print("beta in main", weight)
    weight, n = grad_descent(X, y, weight, 0.01, 0.005)
    predict = predict_class(weight, X)
    # print(len(predict))
    # print("predict",predict)
    # print("the computed theta value is", weight)
    # print("difference between y and predicted y")
    # print("The acutal output for the given input data is where '1' represents male and '0' representd female", y)
    # print("The predicted output for the given data is", predict)
    # print("len of y",len(y))

    y_count = np.sum(y == predict)

    Accuracy = (y_count / len(y)) * 100
    # print("The calculated regression coefficients are", weight)
    # print("total no of iterations taken", n)
    # print("The count of the labels predicted", y_count)
    # print("Accuracy is",(y_count/len(y))*100)
    # print("y out",y)
    # fit(x,y,1)
    # acc.append(Accuracy)
    # plot_reg(X, y, weight)
    return X, y, weight, n




class Adaboost(object):

    def __init__(self, X_train, y_train):
        self.X_train = X_train

        self.N = self.X_train.shape[0]

        self.t_epsilon = []

        self.hist = {}
        self.y_train = y_train
        self.weights = np.ones(self.N) / self.N
        self.alphas = []
        self.clfs = []
        # self.num_estimators = T

    def boosting_algo(self):
        #         for t in range(self.num_estimators):
        for t in range(100):
            #print("t value is ",t)
            output = np.random.choice(self.N, self.N, p=self.weights)

            for t in output:
                self.hist[t] = self.hist.get(t, 0) + 1

            #             print output
            #print("output is",output)
            Xtrain_bst = self.X_train[output]
            #print("Xtrain_bst is",Xtrain_bst)

            ytrain_bst = self.y_train[output]

            X,y,weight,n = logistic_regression(Xtrain_bst, ytrain_bst)
            #             print ls.weights
            #print("comes here")
            #print("x b",X)
            #self.X_train = X
            Y_pred = predict_class(weight,self.X_train)



            error_rate = np.sum((Y_pred != self.y_train) * self.weights)

            if error_rate > 0.5:
                weight = -weight
                Y_pred = predict_class(weight,self.X_train)

            #             print error_rate
            self.t_epsilon.append(error_rate)


            self.clfs.append(weight)
            alpha_t = 0.5 * np.log((1 - error_rate) / error_rate)
            self.alphas.append(alpha_t)


            #             print alpha_t
            self.weights = self.weights * np.exp(-alpha_t * Y_pred * self.y_train)
            self.weights = self.weights / np.sum(self.weights)


X_train,y_train,X_test,y_test = data_clean1()

def x_append(data):
    ones = np.ones((data.shape[0],1))
    data = np.hstack((ones, data))
    return data





X_train = x_append(X_train)
X_test = x_append(X_test)

training_error = []
testing_error = []

boost = Adaboost(X_train, y_train)
#print("x train in boosting",X_train)
boost.boosting_algo()

training_error1 = []
testing_error1 = []
for t in tqdm(range(1, 2)):
    #print("the value of t is ",t)
    tot_train = np.zeros(X_train.shape[0])
    tot_test = np.zeros(X_test.shape[0])
    for i in range(t):
        alpha = boost.alphas[i]
        classifier = boost.clfs[i]
        #             print X_train.shape
        #             print classifier.weights.shape
        #             print np.dot(X_train, classifier.weights).shape
        #print("classifier",classifier[0])
        tot_train += (alpha * predict_class(classifier[0],X_train))
        tot_test += (alpha * predict_class(classifier[0],X_test))
    s_train_pred = np.sign(tot_train)
    s_test_pred = np.sign(tot_test)

    training_error1.append(np.sum(s_train_pred != y_train) / y_train.shape[0])
    testing_error1.append(np.sum(s_test_pred != y_test) / y_test.shape[0])
print("training_error is ",training_error1)
print("testing_error is ",testing_error1)
#plotting the training error and testing error for values 1 to 50
plt.figure()
plt.scatter(1, training_error1, label="Training error")
plt.scatter(1, testing_error1, label="Testing error")
plt.title("Training and testing without iteration")
plt.legend()
plt.show()

plt.close()


