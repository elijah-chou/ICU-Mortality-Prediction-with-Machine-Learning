from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
'''
Splitting the data into training set and testing set with test_size=0.3
Normalizing the xTest and xTrain with standardScaler
'''
def main():
    data=pd.read_csv('extracted_features.csv')
    X=data.drop(columns=['hospital_expire_flag'])
    y=data['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train=X_train.iloc[:,1:]
    X_test=X_test.iloc[:,1:]
    #normalize data using standard scaler
    xTrainEth=X_train['ethnicity']
    xTestEth=X_test['ethnicity']
    X_train=X_train.drop(columns=['ethnicity'])
    X_test=X_test.drop(columns=['ethnicity'])
    xTrainCol=X_train.columns
    xTestCol=X_test.columns
    standard_scaler = preprocessing.StandardScaler()
    xTrain = pd.DataFrame(standard_scaler.fit_transform(X_train),index=X_train.index, columns=xTrainCol)
    xTest = pd.DataFrame(standard_scaler.transform(X_test),index=X_test.index,columns=xTestCol)
    xTrain=pd.concat([xTrain, xTrainEth], axis=1)
    xTest=pd.concat([xTest, xTestEth], axis=1)
    print(xTrain)
    print(xTest)
    pd.DataFrame(xTrain).to_csv('xTrain.csv')
    pd.DataFrame(xTest).to_csv('xTest.csv')
    pd.DataFrame(y_train).to_csv('yTrain.csv')
    pd.DataFrame(y_test).to_csv('yTest.csv')


if __name__ == "__main__":
    main()