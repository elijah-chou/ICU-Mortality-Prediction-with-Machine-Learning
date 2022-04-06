import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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

def k_fold_CV(xFeat, y):
    kf=KFold(n_splits=5)
    data = []
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat.iloc[train_index], xFeat.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]
        data.append([xTrain, xTest, yTrain, yTest])
    return data

def logisticReg(xTrain, yTrain):
    model=LogisticRegression(max_iter=400)
    model.fit(xTrain, yTrain)
    return model

def naiveB(xTrain, yTrain):
    mnb=GaussianNB()
    #mnb.fit(xTrain, yTrain.values.ravel())
    mnb.fit(xTrain, yTrain)
    return mnb

'''
Input: model: single sklearn model with prediction method, xTest, yTest
Output: fpr, tpr
''' 
def model_predict(model, xTest, yTest):
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    return fpr, tpr

'''
model_set: dictionary object where keys are models' names and values are models
output: acc: dictionary object where keys are models' names and values are models' accuracy in predicting test data
        auc: dictionary object where keys are models' names and values are models' AUC in predicting test data
'''
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
    return acc, auc

'''
Ensemble model: For each sample, take the mode of the prediction by all previous samples
'''    
def ensemble(model_set, xTest):
    yHat=np.zeros(len(xTest))
    for key in model_set:
        yHatPredict=model_set[key].predict(xTest)
        yHat=np.add(yHat, yHatPredict)
    for row in range(len(yHat)):
        if yHat[row] < len(model_set)/2:
            yHat[row]=0
        else:
            yHat[row]=1
    return yHat


def main():
    # load the train and test data
    xTrain=pd.read_csv('xTrain.csv')
    xTest=pd.read_csv('xTest.csv')
    yTrain=pd.read_csv('yTrain.csv')
    yTest=pd.read_csv('yTest.csv')
    xTrain=xTrain.drop(columns=['ethnicity'])
    xTest=xTest.drop(columns=['ethnicity'])
    yTrain=yTrain['hospital_expire_flag']
    yTest=yTest['hospital_expire_flag']
    model={}
    model['Gaussian Naive Bayes']=naiveB(xTrain, yTrain)
    model['Logistic Regression']=logisticReg(xTrain, yTrain)
    model['Decision Tree']=decisionTree(xTrain, yTrain, 'entropy', 10, 100)
    model['KNN']=knn(xTrain, yTrain, 10)
    model['Random Forest']=randomForest(xTrain, yTrain, 400, 'entropy', 20, 50, 17, True, True)
    acc, auc=model_estimate(model, xTest, yTest)
    acc['ensemble']=accuracy_score(ensemble(model, xTest),yTest)
    print(acc)
    print(auc)
    


if __name__ == "__main__":
    main()
