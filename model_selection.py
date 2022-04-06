import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def decisionTree(xTrain, yTrain, cri, md, msl):
    model=DecisionTreeClassifier(criterion = cri, max_depth = md, min_samples_leaf = msl)
    model.fit(xTrain, yTrain)
    return model

def knn(xTrain, yTrain, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrain, yTrain)
    return model

def randomForest(xTrain, yTrain, nTree, cri, md, msl, mf, boot, oob):
    model = RandomForestClassifier(n_estimators = nTree, criterion = cri, max_depth = md, min_samples_leaf = msl, max_features = mf, bootstrap = boot, oob_score = oob)
    model.fit(xTrain, yTrain)
    return model

def logisticReg(xTrain, yTrain):
    model=LogisticRegression()
    model.fit(xTrain, yTrain)
    return model

def naiveB(xTrain, yTrain):
    mnb=GaussianNB()
    mnb.fit(xTrain, yTrain.values.ravel())
    return mnb

def model_predict(model, xTest, yTest):
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    return fpr, tpr

def model_estimate(model_set, xTest, yTest):
    acc={}
    auc={}
    for key in model_set:
        fpr, tpr= model_predict(model_set[key], xTest, yTest)
        plt.plot(fpr,tpr,label=key)
        plt.legend()
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        testAuc = metrics.auc(fpr, tpr)
        auc[key]=testAuc
        yHatTest = model_set[key].predict(xTest)
        testAcc = accuracy_score(yTest, yHatTest)
        acc[key]=testAcc
    plt.show()


def k_fold_CV(xFeat, y):
    kf=KFold(n_splits=5)
    data = []
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat.iloc[train_index], xFeat.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]
        data.append([xTrain, xTest, yTrain, yTest])
    return data

def main():
    # load the train and test data
    xTrain=pd.read_csv('xTrain.csv')
    xTest=pd.read_csv('xTest.csv')
    yTrain=pd.read_csv('yTrain.csv')
    yTest=pd.read_csv('yTest.csv')
    xTrain=xTrain.iloc[:, 1:]
    xTest=xTest.iloc[:, 1:]
    yTrain=yTrain.iloc[:, 1]
    yTest=yTest.iloc[:, 1]

    #exclude race features
    #xTrain_no_race = xTrain.drop(columns=["ethnicity", "ASIAN", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "WHITE"])
    xTrain_no_race = xTrain.drop(columns=["ethnicity"])
    #select parameter
    #1. prepare dataset

    #2. select parameters for different models one by one

    #a. knn
    #parameter: k: [5, 10, 50, 100, 500, 1000]
    '''
    avgAcc = []
    n = [5, 10, 50, 100, 500, 1000]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = knn(kxTrain, kyTrain, i)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When k = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best k for knn is: ", n[max_index])


    #b. decision tree
    #parameter: criterion = ["gini", "entropy"]
    avgAcc = []
    n = ["gini", "entropy"]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = decisionTree(kxTrain, kyTrain, i, 15, 50)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When criterion = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best criterion for decision tree is: ", n[max_index])

    #parameter: max_depth = [5, 10, 15, 20, 30]
    avgAcc = []
    n = [5, 10, 15, 20, 30]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = decisionTree(kxTrain, kyTrain, "gini", i, 50)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When maximum depth = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best maxDepth for decision tree is: ", n[max_index])

    #parameter: min_samples_leaf = [10, 50, 100, 500, 1000]
    avgAcc = []
    n = [10, 50, 100, 500, 1000]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = decisionTree(kxTrain, kyTrain, "gini", 15, i)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When min_samples_leaf = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best min_samples_leaf for decision tree is: ", n[max_index])

    #c. random forest
    #parameter: number of trees = [10, 50, 100, 200, 400]
    avgAcc = []
    n = [10, 50, 100, 200, 400]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, i, "gini", 10, 30, len(kxTrain.iloc[0]), True, True)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When number of trees = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best number of trees for randomForest is: ", n[max_index])

    #parameter: criterion = ["gini", "entropy"]
    avgAcc = []
    n = ["gini", "entropy"]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, 400, i, 10, 30, len(kxTrain.iloc[0]), True, True)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When criterion = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best criterion for randomForest is: ", n[max_index])
    '''
    #parameter: maximum depth = [5, 10, 15, 20, 30]
    avgAcc = []
    n = [5, 10, 15, 20, 30]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, 400, "entropy", i, 30, len(kxTrain.iloc[0]), True, True)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When maximum depth = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best maxDepth for randomForest is: ", n[max_index])

    #parameter: minimum leaf = [10, 50, 100, 500, 1000]
    avgAcc = []
    n = [10, 50, 100, 500, 1000]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, 400, "entropy", 10, i, len(kxTrain.iloc[0]), True, True)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When min_samples_leaf = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best min_samples_leaf for randomForest is: ", n[max_index])

    #parameter: max number of features = loop through all features
    avgAcc = []
    n = len(xTrain_no_race.iloc[0])
    for i in range(1, n):
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, 400, "entropy", 10, 30, i, True, True)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When max_features = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best max_features for randomForest is: ", max_index)
    
    #parameter: bootstrap or not = [True, False]

    avgAcc = []
    n = [True, False]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            #since oob can't = True when bootstrap = False, so we just set them the same
            model = randomForest(kxTrain, kyTrain, 400, "entropy", 10, 30, len(kxTrain.iloc[0]), i, i)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When bootstrap = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best bootstrap for randomForest is: ", n[max_index])

    #parameter: using out-of-bag samples or not = [True, False]
    avgAcc = []
    n = [True, False]
    for i in n:
        acc = 0.0
        for j in range(5):
            kxTrain, kxTest, kyTrain, kyTest = k_fold_CV(xTrain_no_race, yTrain)[j]
            model = randomForest(kxTrain, kyTrain, 400, "entropy", 10, 30, len(kxTrain.iloc[0]), True, i)
            y_pred = model.predict(kxTest)
            acc += accuracy_score(kyTest, y_pred)
        avgAcc.append(acc/5)
        print("When oob = ", i, "the average accuracy score for 5-fold is: ", acc/5)
    max_value = max(avgAcc)
    max_index = avgAcc.index(max_value)
    print("The best oob for randomForest is: ", n[max_index])


if __name__ == "__main__":
    main()
