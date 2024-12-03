import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
# import hiplot as hip

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


st.set_page_config(page_title="Data Science: SMOTE")

st.markdown("# Data Science: SMOTE")
st.sidebar.header("Data Science: SMOTE")
st.write(
    """
    This page explores SMOTE on the second dataset (See the Health Data page)
    """
)

###Importing the data
# Load in the physical activity specific data
shl_url = 'https://raw.githubusercontent.com/jansena4/CMSE830/refs/heads/main/Sleep_health_and_lifestyle_dataset.csv'
shl = pd.read_csv(shl_url)


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


###Plots, etc.
#Introduced missing values
shl_missing = shl.copy() #Making a copy of the dataframe to add missing values to
shl_mask = np.random.rand(shl_missing.shape[0]) < 0.4  #Making a mask to remove random values
shl_missing.loc[shl_mask, 'Age'] = np.nan #Removing the masked values


### SimpleImputer Imputation - mean
my_imp = SimpleImputer(missing_values=np.NaN)

# impute after temporarily removing the methods column
X = shl_missing.drop(columns=["Occupation", "Blood Pressure",  "Sleep Disorder"])
imputed_X = my_imp.fit_transform(X)

# create DataFrame with correct column names and index
fixed_X = pd.DataFrame(imputed_X, columns=X.columns, index=X.index)

# add the 'method' column back
shl_imp = pd.concat([fixed_X, shl_missing[["Occupation", "Blood Pressure",  "Sleep Disorder"]]], axis=1)

# ensure the columns are in the same order as the original DataFrame
shl_imp = shl_imp[shl_missing.columns]

sns.heatmap(shl_imp.isna().transpose(), cmap="plasma")
plt.show()

sns.histplot(shl_missing['Age'], bins = len(np.unique(shl_missing['Age'])), color='red', kde=True)
sns.histplot(shl_imp['Age'], bins = len(np.unique(shl_imp['Age'])), color='blue', kde=True)


### SMOTE
X = shl_imp.drop(['Sleep Disorder', 'Occupation', 'Blood Pressure'], axis=1)
y = shl_imp['Sleep Disorder'].astype(int)

# display original class distribution
print("\nOriginal class distribution:")
print(y.value_counts())

# apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_shl = pd.DataFrame(X_resampled, columns=X.columns)
resampled_shl['Sleep Disorder'] = y_resampled



# Bar chart for class distribution after SMOTE
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x=y)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Sleep Disorder')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.countplot(x=y_resampled)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Sleep Disorder')
plt.ylabel('Count')
st.pyplot(plt)


plt.figure(figsize=(10,10))

plt.subplot(4,2,1)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Age'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Age'], color='red', label='Sleep Disorder', kde=True)
plt.title('Age Distribution Before SMOTE')
plt.legend()

plt.subplot(4,2,2)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Age'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Age'], color='red', label='Sleep Disorder', kde=True)
plt.title('Age Distribution After SMOTE')
plt.legend()


plt.subplot(4,2,3)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Gender'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Gender'], color='red', label='Sleep Disorder', kde=True)
plt.title('Gender Distribution Before SMOTE')
plt.legend()

plt.subplot(4,2,4)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Gender'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Gender'], color='red', label='Sleep Disorder', kde=True)
plt.title('Gender Distribution After SMOTE')
plt.legend()


plt.subplot(4,2,5)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Quality of Sleep'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Quality of Sleep'], color='red', label='Sleep Disorder', kde=True)
plt.title('Sleep Quality Distribution Before SMOTE')
plt.legend()

plt.subplot(4,2,6)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Quality of Sleep'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Quality of Sleep'], color='red', label='Sleep Disorder', kde=True)
plt.title('Sleep Quality Distribution After SMOTE')
plt.legend()


plt.subplot(4,2,7)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Daily Steps'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Daily Steps'], color='red', label='Sleep Disorder', kde=True)
plt.title('Daily Steps Distribution Before SMOTE')
plt.legend()

plt.subplot(4,2,8)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Daily Steps'], color='blue', label='No Sleep Disorder', kde=True)
sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Daily Steps'], color='red', label='Sleep Disorder', kde=True)
plt.title('Daily Steps Distribution After SMOTE')
plt.legend()


plt.tight_layout()
st.pyplot(plt)




