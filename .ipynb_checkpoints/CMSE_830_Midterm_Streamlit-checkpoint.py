import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to the CMSE 830 Midterm Project! 👋")

st.sidebar.success("Please select a page above")

st.write("""
    This project hopes to explore the relationships between physical activity levels, general health measures, and sleep patterns.

    There are several ideas for what to check:
    - The relationship between ages, their bedtime and wake-up time, and the overall sleep quality. This would be interesting to explore the stereotype that teenagers and young adults go to bed later than adults and therefore get worse sleep.
    - The relationships between physical activity levels, sleep, and overall health. Potentially comparing measures of general health against physical activity measure and sleep duration/quality to determine which measure is more predictive of health levels.
    - General trends among these factors (age, gender, bmi level, etc.) and activity level. This would be an interesting way to see whetehr the stereotype of "gym bros" working out more/being healthier than the general population seems to be true.



    The GitHub repository for this project can be found here: https://github.com/jansena4/CMSE830
    """)




