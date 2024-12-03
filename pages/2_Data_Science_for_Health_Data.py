import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
# import hiplot as hip
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


st.set_page_config(page_title="Data Science for Health Data")

st.markdown("# Data Science for Health Data")
st.sidebar.header("Data Science for Health Data")
st.write(
    """
    This dataset was found in kaggle here: https://www.kaggle.com/datasets/hanaksoy/health-and-sleep-statistics


    This is the second dataset for our analysis, it focuses more on the health aspects of the sleep and health data. This includes various aspects such as bmi levels, physical activity level, daily steps, heart rate, and stress level.
"""
)

###Importing the data
# Load in the physical activity specific data
shl_url = 'https://raw.githubusercontent.com/jansena4/CMSE830/refs/heads/main/Sleep_health_and_lifestyle_dataset.csv'
shl = pd.read_csv(shl_url)

# Display the physical activity specific data
st.write("#### Physical Activity Specific Data")
st.write(shl)



###Data Cleaning
#Converting variable types
shl = shl.drop(columns=['Person ID'])

shl.loc[shl['Gender']=='Female', 'Gender'] = 0 #Setting female as 0
shl.loc[shl['Gender']=='Male', 'Gender'] = 1 #Setting male as 1

shl.loc[shl['Sleep Disorder']=='Sleep Apnea', 'Sleep Disorder'] = 1 #Setting yes sleep disorder as 1
shl.loc[shl['Sleep Disorder']=='Insomnia', 'Sleep Disorder'] = 1 #Setting yes sleep disorder as 1
shl.loc[shl['Sleep Disorder'].isnull(), 'Sleep Disorder'] = 0 #Setting no sleep disorder as 1

shl.loc[shl['BMI Category']=='Normal Weight', 'BMI Category'] = 0 #Setting normal weight bmi as 0
shl.loc[shl['BMI Category']=='Normal', 'BMI Category'] = 0 #Setting normal bmi as 0
shl.loc[shl['BMI Category']=='Overweight', 'BMI Category'] = 1 #Setting overweight bmi as 1
shl.loc[shl['BMI Category']=='Obese', 'BMI Category'] = 2 #Setting obese bmi as 2

st.write("#### Cleaned & Transformed Physical Activity Specific Data")
st.write(shl)


###Plots, etc.
#Correlation heatmap
st.write(f"#### Correlation Heatmap")
plt.figure(figsize=(8, 6))
shl_encoded = pd.get_dummies(shl) # Turn all categorical variables into one-hot encoded dummy variables
sns.heatmap(shl_encoded.corr(), cmap="plasma") #Plot a heatmap for the correlations between all variables (including dummy variables)
st.pyplot(plt)
st.write("This heatmap shows that most variables have little to no correlation in the data")

#Introduced missing values
shl_missing = shl.copy() #Making a copy of the dataframe to add missing values to
shl_mask = np.random.rand(shl_missing.shape[0]) < 0.4  #Making a mask to remove random values
shl_missing.loc[shl_mask, 'Age'] = np.nan #Removing the masked values

st.write(f"#### Heatmap of Missing Values")
plt.figure(figsize=(8, 6))
sns.heatmap(shl_missing.isna().transpose(), cmap="plasma") #Showing a heatmap of the now missing values
st.pyplot(plt)
st.write("Here we introduced some missing values to the Age in the dataset for the purpose of exploring imputation methods.")


### Dropbox
st.write("#### Variable Plots")

option = st.selectbox(
    'Select a variable to explore:',
    shl.columns
)
st.write('You selected:', option)

for col in shl.columns:
    if (col == option) & (col not in ['Occupation', 'Blood Pressure']):
        st.write(f"#### Interactive Histogram of {col}")
        hist_values = np.histogram(shl[col], bins=len(set(shl[col])))[0]
        st.bar_chart(hist_values)

        st.write(f"#### Histogram & KDE of {col}")
        plt.figure(figsize=(8, 6))
        sns.histplot(shl[col], bins=len(set(shl[col])), color='blue', kde=True)
        st.pyplot(plt)


        # Add plots to compare variables
        st.write("Variable Comparison Plots")
        option_2 = st.selectbox(
            'Select a second variable to explore:',
            set(shl.columns) - set([option, 'Occupation', 'Blood Pressure'])
        )
        st.write('You selected:', option_2)

        st.write(f"#### Plot of {col} vs {option_2}")
        plt.figure(figsize=(8, 6))
        plt.scatter(shl[col], shl[option_2], color='blue')
        plt.xlabel(f'{col}')
        plt.ylabel(f'{option_2}')
        plt.title(f'Plot of {col} vs {option_2}')
        st.pyplot(plt)

        option_col = st.selectbox(
            'Select another variable as the color for the plot:',
            set(shl.columns) - set([option, option_2, 'Occupation', 'Blood Pressure'])
        )
        st.write('You selected:', option_col)
        fig = px.scatter(shl, x=col, y=option_2, color=option_col, title='Interactive Scatter Plot')
        st.plotly_chart(fig)

    if (col == option) & (col in ['Occupation', 'Blood Pressure']):
        st.write("You've chosen a categorical variable, the category are listed below")
        st.write(f'Choices in {col}', set(shl[col]))





