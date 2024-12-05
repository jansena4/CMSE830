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


st.set_page_config(page_title="Sleep Data")

st.markdown("# Sleep Data")
st.sidebar.header("Sleep Data")
st.write(
    """
    This dataset was found in kaggle here: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset


    This is the first dataset for our analysis, it focuses more on the sleep aspects of the sleep and health data. This includes various aspects such as bedtime, wake-up time, sleep disorders, and potential medications.
"""
)

###Importing the data
# Load in the sleep specific data
hss_url = 'https://raw.githubusercontent.com/jansena4/CMSE830/refs/heads/main/Health_Sleep_Statistics.csv'
hss = pd.read_csv(hss_url)

# Display the sleep specific data
st.write("#### Original Sleep Specific Data")
st.write(hss)



# ##Data Cleaning
# Converting variable types
# hss['Bedtime'] = pd.to_datetime(hss['Bedtime'], format='%H:%M')#.dt.time #Converting bedtimes to datetimes
# hss['Wake-up Time'] = pd.to_datetime(hss['Wake-up Time'], format='%H:%M')#.dt.time #Converting wake-up times to datetimes

hss = hss.drop(columns=['User ID'])

hss.loc[hss['Physical Activity Level']=='low', 'Physical Activity Level'] = -1 #Setting low physical activity level as -1
hss.loc[hss['Physical Activity Level']=='medium', 'Physical Activity Level'] = 0 #Setting medium physical activity level as 0
hss.loc[hss['Physical Activity Level']=='high', 'Physical Activity Level'] = 1 #Setting high physical activity level as 1

hss.loc[hss['Dietary Habits']=='unhealthy', 'Dietary Habits'] = -1 #Setting unhealthy dietary habits as -1
hss.loc[hss['Dietary Habits']=='medium', 'Dietary Habits'] = 0 #Setting unhealthy dietary habits as 0
hss.loc[hss['Dietary Habits']=='healthy', 'Dietary Habits'] = 1 #Setting healthy dietary habits as 1

hss.loc[hss['Sleep Disorders']=='no', 'Sleep Disorders'] = 0 #Setting no sleep disorder as 0
hss.loc[hss['Sleep Disorders']=='yes', 'Sleep Disorders'] = 1 #Setting yes sleep disorder as 1

hss.loc[hss['Medication Usage']=='no', 'Medication Usage'] = 0 #Setting no medication usage as 0
hss.loc[hss['Medication Usage']=='yes', 'Medication Usage'] = 1 #Setting yes medication usage as 1

hss.loc[hss['Gender']=='f', 'Gender'] = 0 #Setting female as 0
hss.loc[hss['Gender']=='m', 'Gender'] = 1 #Setting male as 1

st.write("#### Cleaned & Transformed Sleep Specific Data")
st.write(hss)

###Plots, etc.
#Correlation heatmap
st.write(f"#### Correlation Heatmap")
plt.figure(figsize=(8, 6))
hss_encoded = pd.get_dummies(hss) # Turn all categorical variables into one-hot encoded dummy variables
sns.heatmap(hss_encoded.corr(), cmap="plasma") #Plot a heatmap for the correlations between all variables (including dummy variables)
st.pyplot(plt)
st.write("This heatmap shows that most variables have little to no correlation in the data")

###Dropbox
st.write("#### Variable Plots")

option = st.selectbox(
    'Select a variable to explore:',
    hss.columns
)
st.write('You selected:', option)

for col in hss.columns:
    if (col == option) & (col not in ['Bedtime', 'Wake-up Time']):
        st.write(f"#### Interactive Histogram of {col}")
        hist_values = np.histogram(hss[col], bins=len(set(hss[col])))[0]
        st.bar_chart(hist_values)

        st.write(f"#### Histogram & KDE of {col}")
        plt.figure(figsize=(8, 6))
        sns.histplot(hss[col], bins=len(set(hss[col])), color='blue', kde=True)
        st.pyplot(plt)


        # Add plots to compare variables
        st.write("Variable Comparison Plots")
        option_2 = st.selectbox(
            'Select a second variable to explore:',
            set(hss.columns) - set([option, 'Bedtime', 'Wake-up Time'])
        )
        st.write('You selected:', option_2)

        st.write(f"#### Plot of {col} vs {option_2}")
        plt.figure(figsize=(8, 6))
        plt.scatter(hss[col], hss[option_2], color='blue')
        plt.xlabel(f'{col}')
        plt.ylabel(f'{option_2}')
        plt.title(f'Plot of {col} vs {option_2}')
        st.pyplot(plt)

        option_col = st.selectbox(
            'Select another variable as the color for the plot:',
            set(hss.columns) - set([option, option_2, 'Bedtime', 'Wake-up Time'])
        )
        st.write('You selected:', option_col)
        fig = px.scatter(hss, x=col, y=option_2, color=option_col, title='Interactive Scatter Plot')
        st.plotly_chart(fig)

    if (col == option) & (col in ['Bedtime', 'Wake-up Time']):
        st.write("You've chosen a categorical variable, the category are listed below")
        st.write(f'Choices in {col}', set(hss[col]))





