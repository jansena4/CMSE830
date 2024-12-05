# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(
    page_title="Sleep, Health, and Physical Activity",
    page_icon="👋",
)

st.write("# Sleep, Health, and Physical Activity")

option = st.selectbox(
    'Please select a page:',
    ['Welcome', 'Sleep Data', 'Health Data', 'College Sleep Data', 'SMOTE', 'Modeling']#, 'PIY (Predict It Yourself!)']
)

# st.write('Now viewing:', option)

if option == 'Welcome':


    st.write("## Welcome to the future of sleep and health! 👋")
    
    # st.sidebar.success("Please select a page above") #wouldn't work (not allowed to cry about this though)
    
    st.write("""
        This analysis hopes to explore the relationships between physical activity levels, general health measures, and sleep patterns.
    
        There are several ideas for what to check:
        - The relationship between ages, their bedtime and wake-up time, and the overall sleep quality. This would be interesting to explore the stereotype that teenagers and young adults go to bed later than adults and therefore get worse sleep.
        - The relationships between physical activity levels, sleep, and overall health. Potentially comparing measures of general health against physical activity measure and sleep duration/quality to determine which measure is more predictive of health levels.
        - General trends among these factors (age, gender, bmi level, etc.) and activity level. This would be an interesting way to see whetehr the stereotype of "gym bros" working out more/being healthier than the general population seems to be true.
    
    
        Please see the tabs to the left to explore each dataset
    
    
    
        The GitHub repository for this analysis can be found here: https://github.com/jansena4/CMSE830
        """)


if option == 'Sleep Data':

    st.write("# Sleep Data")
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
    st.write("Below we have the new dataset with cleaned data. This data had several things done to it including turning the sleep disorder, medication usage, and gender features into binary features, as well as ranking low/medium/high activity levels and unhealthy/medium/healthy diets as -1/0/1 (creating ordinal features).")
    st.write(hss)
    
    ###Plots, etc.
    #Correlation heatmap
    st.write(f"#### Correlation Heatmap")
    st.write(f"Below we have a correlation heatmap of all of the numerical features.")
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
    st.write(f"The first plot shown is the distribution of {option} from the data. This plot helps us to understand what the data looks like going into our analysis and model.")
    
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
            st.write(f"The next plot we examine is a plot of the relationship between {option} (on the x axis) and {option_2} (on the y axis). This helps us see correlations or non0linear relationships.")
    
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
            st.write(f"The last plot we'll examine is a plot of {option} vs {option_2} where the {option_col} feature is represented by the colors in the plot.")
            fig = px.scatter(hss, x=col, y=option_2, color=option_col, title='Interactive Scatter Plot')
            st.plotly_chart(fig)
    
        if (col == option) & (col in ['Bedtime', 'Wake-up Time']):
            st.write("You've chosen a categorical variable, the category are listed below")
            st.write(f'Choices in {col}', set(hss[col]))















if option == 'Health Data':

    st.write("## Health Data")
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
    st.write("Below we have the new dataset with cleaned data. This data had several things done to it including turning the sleep disorder feature from a category of Sleep Apnea, Insomnia, or none into a binary feature for the presence of sleep disorders, and turning the four BMI categories into an ordinal feature.")
    st.write(shl)
    
    
    ###Plots, etc.
    #Correlation heatmap
    st.write(f"#### Correlation Heatmap")
    st.write(f"Below we have a correlation heatmap of all of the numerical features.")
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
    st.write(f"The first plot shown is the distribution of {option} from the data. This plot helps us to understand what the data looks like going into our analysis and model.")
    
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
            st.write(f"The next plot we examine is a plot of the relationship between {option} (on the x axis) and {option_2} (on the y axis). This helps us see correlations or non0linear relationships.")
    
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
            st.write(f"The last plot we'll examine is a plot of {option} vs {option_2} where the {option_col} feature is represented by the colors in the plot.")
            
            fig = px.scatter(shl, x=col, y=option_2, color=option_col, title='Interactive Scatter Plot')
            st.plotly_chart(fig)
    
        if (col == option) & (col in ['Occupation', 'Blood Pressure']):
            st.write("You've chosen a categorical variable, the category are listed below")
            st.write(f'Choices in {col}', set(shl[col]))
















if option == 'College Sleep Data':
    st.write("## College Sleep Data")















if option == 'SMOTE':

    st.write("## SMOTE")
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

    st.write(
        """
        The heatmap below shows the indices of the missing values in the dataset.
        """
    )
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
    
    
    st.write("Next we're going to explore the distribution of the data before and after we impute the missing values. The first plots look slightly different due to their categorical (rather than numerical) nature.")
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
    sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Daily Steps'], color='blue', label='No Sleep Disorder', kde=True)
    sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Daily Steps'], color='red', label='Sleep Disorder', kde=True)
    plt.title('Daily Steps Distribution Before SMOTE')
    plt.legend()
    
    plt.subplot(4,2,6)
    sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Daily Steps'], color='blue', label='No Sleep Disorder', kde=True)
    sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Daily Steps'], color='red', label='Sleep Disorder', kde=True)
    plt.title('Daily Steps Distribution After SMOTE')
    plt.legend()


    st.write("The last plots we look at will be our target variable, sleep quality.")

    plt.subplot(4,2,7)
    sns.histplot(shl_imp[shl_imp['Sleep Disorder']==0]['Quality of Sleep'], color='blue', label='No Sleep Disorder', kde=True)
    sns.histplot(shl_imp[shl_imp['Sleep Disorder']==1]['Quality of Sleep'], color='red', label='Sleep Disorder', kde=True)
    plt.title('Sleep Quality Distribution Before SMOTE')
    plt.legend()
    
    plt.subplot(4,2,8)
    sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==0]['Quality of Sleep'], color='blue', label='No Sleep Disorder', kde=True)
    sns.histplot(resampled_shl[resampled_shl['Sleep Disorder']==1]['Quality of Sleep'], color='red', label='Sleep Disorder', kde=True)
    plt.title('Sleep Quality Distribution After SMOTE')
    plt.legend()
    
    
    plt.tight_layout()
    st.pyplot(plt)


    st.write("From these plots we can tell whether the imputations are maintaing the original ditributions of the variables or not.")


















if option == 'Modeling':

    #Loading the data
    hss_url = 'https://raw.githubusercontent.com/jansena4/CMSE830/refs/heads/main/Health_Sleep_Statistics.csv'
    hss = pd.read_csv(hss_url)
    
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




    
    st.write("## Modeling")

    st.write("The first model we will exmaine is a linear regression model. We're going to narrow in on Age, Gender, Sleep Disorders, & Medication Usage from the sleep data to focus our analysis on the variables we're most interested in.")    
    # X = hss[~'Sleep Quality']
    y = hss['Sleep Quality']
    x_cols = ['Age', 'Gender', 'Sleep Disorders', 'Medication Usage']
    X = hss[x_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    y_pred_linreg = lin_reg.predict(X_test)
    
    st.write("Intercept:", lin_reg.intercept_)
    for i in range(len(x_cols)):
        st.write(f"{x_cols[i]} Coefficient:", lin_reg.coef_[i])
    
    mse = mean_squared_error(y_test, y_pred_linreg)
    r2 = r2_score(y_test, y_pred_linreg)
    st.write("Mean Squared Error:", mse)
    st.write("R^2 Score:", r2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_linreg, color='blue')
    
    plot_line = np.linspace(int(np.min([np.min(y_test), np.min(y_pred_linreg)])),
                           int(np.max([np.max(y_test), np.max(y_pred_linreg)])))
    plt.plot(plot_line, plot_line, color = 'red')
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Linear Regression")
    plt.grid(True)
    plt.show()







    
    st.write('#### Please Select values below')
    model_to_use = st.selectbox('Predict with model:', ['Linear Regression', 'KNN', 'Random Forest'])

    # x_vals = {}
    age = st.slider('Select age:', 0, 100, 25)
    gender = st.selectbox('Select Gender:', ['Male', 'Female'])
    sleep_dis = st.selectbox('Select Sleep Disorder:', ['Yes', 'No'])
    med_use = st.selectbox('Select Medication Usage:', ['Yes', 'No'])

    
    if gender == 'Male':
        # x_vals['Gender'] = 1
        gender_val = 1

    if gender == 'Female':
        # x_vals['Gender'] = 0
        gender_val = 0

    
    if sleep_dis == 'Yes':
        # x_vals['Sleep Disorders'] = 1
        sleepdis_val = 1

    if sleep_dis == 'No':
        # x_vals['Sleep Disorders'] = 0
        sleepdis_val = 0

    
    if med_use == 'Yes':
        # x_vals['Medication Usage'] = 1
        meduse_val = 1

    if med_use == 'No':
        # x_vals['Medication Usage'] = 0
        meduse_val = 0

    
    
    X_vals = pd.DataFrame([[age, gender_val, sleepdis_val, meduse_val]], columns=x_cols)
    
    if model_to_use == 'Linear Regression':
        sleep_qual = lin_reg.predict(X_vals)
    
    # if model_to_use == 'KNN':
    #     sleep_qual = knn.predict(X_vals)
    
    # if model_to_use == 'Random Forest':
    #     sleep_qual = ran_for.predict(X_vals)
        
    sleep_qual =  x_vals['Age']/10 - x_vals['Gender'] - x_vals['Sleep Disorders'] -  x_vals['Medication Usage']

    st.write(f'We predict you will have a {sleep_qual}/10 sleep!')
    













