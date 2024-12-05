import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to the future of sleep and health! 👋")

st.sidebar.success("Please select a page above")

st.write("""
    Today we're exploring the relationships between physical activity levels, general health measures, and sleep patterns.

    There are several categories of activity, health and sleep to explore:
    - Teenagers: Have the stereotype of going to be at later times and not getting enough sleep while also being more stressed on average. We will explore whether these later hours and stress levels affect their quality of sleep.
    - "Gym Bros": People who tend to go to the gym more often. We will discuss how higher than average activity levels might impact sleep quality.
    - Insomniacs: People who struggle with sleep due to a sleep disorder. We will look into the correlation between sleep duration, time fallen asleep, time woken up, and the presence of a sleep disorder as well as the impact on the quality of sleep.


    To the left you will see several tabs to explore.
    

    The GitHub repository for the data and code behind this exploration can be found here: https://github.com/jansena4/CMSE830
    """)




