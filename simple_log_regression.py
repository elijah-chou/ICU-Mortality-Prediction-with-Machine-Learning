from os import error
import pandas as pd
from sklearn.linear_model import LogisticRegression
def logReg(xTrain, yTrain):
    model=LogisticRegression()
    model.fit(xTrain, yTrain)
    return model

'''
Train a simple log regression model (replicating previous studies) for comparison purpose
'''
def main():
    # load the train and test data
    xTrain=pd.read_csv('xTrain.csv')
    xTest=pd.read_csv('xTest.csv')
    yTrain=pd.read_csv('yTrain.csv')
    yTest=pd.read_csv('yTest.csv')
    yTrain=yTrain['hospital_expire_flag']
    yTest=yTest['hospital_expire_flag']
    xTrainDrop=xTrain['sofa_24hours']
    xTrainDrop=xTrainDrop.values.reshape(-1,1)
    xTestDrop=xTest['sofa_24hours']
    xTestDrop=xTestDrop.values.reshape(-1,1)
    model=logReg(xTrainDrop, yTrain)
    yHat=pd.DataFrame(model.predict(xTestDrop),columns=['hospital_expire_flag'])
    for row in range(len(yHat)):
        if yHat.at[row, 'hospital_expire_flag']>0.5:
            yHat.at[row, 'hospital_expire_flag']=1
        else: 
            yHat.at[row, 'hospital_expire_flag']=0
    errorCount={}
    sampleCount={}
    for row in range(len(yHat)):
        sampleCount[xTest.at[row, 'ethnicity']]=sampleCount.get(xTest.at[row, 'ethnicity'],0)+1
        #set to 1: false positive error; 0: false negative error.
        if yHat.at[row, 'hospital_expire_flag']-yTest.iloc[row]==1:
            errorCount[xTest.at[row, 'ethnicity']]=errorCount.get(xTest.at[row, 'ethnicity'],0)+1
    print ('number of false positive error with respect to race:',errorCount)
    print ('total number of sample with respect to race:', sampleCount)


if __name__ == "__main__":
    main()
