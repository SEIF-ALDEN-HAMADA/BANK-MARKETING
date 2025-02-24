import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from Data_Cleaning import DataCleaningClass

#data reading
data = pd.read_csv("BANK/bank.csv")

# Preview the Data
st.title("BANK DATA DASHBOARD")
st.header("DATA OVER VIEW")
st.write("First Few Rows Of The Data:")
st.dataframe(data.head())
st.write("WHOLE DATA SET")
st.dataframe(data)

#Statistics

st.header("Summary Statistics For Numerical Columns")
st.dataframe(data.describe())

st.header("Summary Statistics ForCategorical Columns")
st.dataframe(data.describe(include=["object"]))

# Columns Selection For Visualization
x_axis = st.selectbox("X-AXIS",data.columns)
y_axis = st.selectbox("Y-AXIS",data.columns)

plot_type=st.radio("Select Plot Type:",["Scatter plot","Line plot","Histogram plot","Bar plot"],index=None)

# Plot Selecting

if plot_type == "Scatter plot":
    st.subheader("Scatter Plot")
    fgr=plt.figure()
    sns.scatterplot(x = x_axis, y = y_axis, data = data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fgr)
elif plot_type == "Line plot":
    st.subheader("Line Plot")
    fgr=plt.figure()
    sns.lineplot(x = x_axis, y = y_axis, data = data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fgr)
elif plot_type == "Histogram plot":
    st.subheader("Histogram Plot")
    fgr=plt.figure()
    sns.histplot(x = x_axis, data = data)
    plt.xlabel(x_axis)
    st.pyplot(fgr)
elif plot_type == "Bar plot":
    st.subheader("Bar Plot")
    fgr=plt.figure()
    sns.barplot(x = x_axis, y = y_axis, data = data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fgr)

st.header("Correlation Heatmap")
if st.button("Generate Heatmap"):
    fgr=plt.figure()
    sns.heatmap(data.corr(numeric_only=True),annot=True,cmap="Blues")
    st.pyplot(fgr)



# Data Distribution Graphs "After Removing Outliers".
st.header("Data Distribution Graphs")
st.write('Data Distribution Graphs "After Removing Outliers".')


Data_info = DataCleaningClass()
num_col = data.select_dtypes("number")
num_col_ol = num_col.drop(columns=["duration","pdays"])
data = Data_info.RemoveOutliers(data,num_col_ol)
data.reset_index(drop=True, inplace= True )


if st.button("Generate HIST PLOT Graphs"):
        fgr=plt.figure(figsize=(20,15))
        for index,col in enumerate(num_col.columns,start=1):
            
            plt.subplot(3,3,index)
            plt.title(f"HIST PLOT {col}")
            sns.histplot(data[col], kde=True)
        plt.show()    
        st.pyplot(fgr)


objectV =data.select_dtypes("object")
data[objectV.columns] = objectV.astype("category")

cat_col =data.select_dtypes("category")

st.header("Categorical-Data Distribution Graphs")
if st.button("Generate COUNT PLOT"):
        fgr=plt.figure(figsize=(20,15))
        for index,col in enumerate(cat_col.columns,start=1):
            plt.subplot(2,4,index)
            plt.title(f"'{col}' COUNT PLOT ")
            sns.countplot(x=data[col])
        
        plt.show()    
        st.pyplot(fgr)
if st.button("Generate PIE PLOT"):
        fgr=plt.figure(figsize=(20,10))
        for index,col in enumerate(cat_col,start=1):
            plt.subplot(2,4,index)
            plt.pie(
                data[col].value_counts(),
                autopct="%1.0f%%",
                labels=data[col].value_counts().index)
        plt.show()    
        st.pyplot(fgr)
st.header("Categorical/Categorical Relationship")
st.write("COUNT PLOT ( Marital / deposit)")
if st.checkbox("Generate COUNT PLOT"):
        fgr = plt.figure()
        sns.countplot(x=data["marital"],hue=data["deposit"])    
        st.pyplot(fgr)

st.header("Numerical / Categorical Relationship")
st.write("BAR PLOT ( Marital / Balance)")
if st.checkbox("Generate BAR PLOT"):
        fgr = plt.figure()
        sns.barplot(x=data["marital"],y=data["balance"],estimator="mean")    
        st.pyplot(fgr)
