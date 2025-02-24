import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataCleaningClass:
    def Data_clean(self,data):
        cols,dtype,nulls,duplicates,uniques = [],[],[],[],[]
        for col in data.columns:
            cols.append(col)
            dtype.append(data[col].dtype)
            nulls.append(data[col].isnull().sum())
            duplicates.append(data.duplicated().sum())
            uniques.append(data[col].nunique())

        data = pd.DataFrame({"Column":cols,"Data Type":dtype,"N-Nulls":nulls,"N-Uniques":uniques,"N-Duplicates":duplicates})
        return data  
    def CheckOutliers(self,data,num_cols):
        plt.figure(figsize=(20,15))
        for index,col in enumerate(num_cols.columns,start=1):
            plt.subplot(3,3,index)
            plt.title(f"BOX PLOT {col}")
            sns.boxplot(x=data[col])
        plt.show()
    def ReplaceOutliers(self,data,num_cols):
        for col in num_cols:
            Q1= data[col].quantile(0.25)
            Q3= data[col].quantile(0.75)
            IQR=Q3-Q1
            lower_fence=Q1-1.5*IQR
            upper_fence=Q3+1.5*IQR
            data[col] = data[col].clip(lower=lower_fence, upper=upper_fence)
        return data
    def RemoveOutliers(self,data,num_cols):
        for col in num_cols:
            Q1= data[col].quantile(0.25)
            Q3= data[col].quantile(0.75)
            IQR=Q3-Q1
            lower_fence=Q1-1.5*IQR
            upper_fence=Q3+1.5*IQR
            data =data[(data[col]>=lower_fence) & (data[col]<= upper_fence)]
        return data