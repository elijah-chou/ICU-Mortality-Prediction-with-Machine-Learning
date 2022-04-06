import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # load the dataset
    data=pd.read_excel('admissions.xlsx')
    #data['admittime']=pd.to_datetime(data['admittime'])
    for row in range(len(data)):
        #update admission_location- 0: not admitted in ER; 1: admitted in ER
        data.at[row, 'admission_location']=int(data.at[row, 'admission_location']=='EMERGENCY ROOM')
        #update insurance info- 0: other type of insurance; 1: medicare
        data.at[row, 'insurance']=int(data.at[row, 'insurance']=='Medicare')
        #update marital_status column- 0: single, widowed, divorced, or unknown; 1: married
        data.at[row,'marital_status']=int(data.at[row,'marital_status']=='MARRIED')
    #drop irrelavent columns to reduce model complexity
    data=data.drop(columns=['subject_id','hadm_id','admittime','dischtime','deathtime','discharge_location','language','edregtime','edouttime','stay_id'])
    data=pd.concat([data,pd.get_dummies(data['admission_type'])],axis=1)
    data=pd.concat([data,pd.get_dummies(data['ethnicity'])],axis=1)
    data=data.drop(columns=['admission_type','BLACK/AFRICAN AMERICAN','ELECTIVE'])

    #shuffle the data 
    data=data.sample(frac=1)
    corPlot=sns.heatmap(data.corr(method='pearson'),annot=True, xticklabels=1,yticklabels=1)
    plt.show()
    #data.to_csv('extracted_features.csv')

    



if __name__ == "__main__":
    main()
