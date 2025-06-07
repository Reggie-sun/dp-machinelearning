import streamlit as st
import pandas as pd

st.title('machine learning app')

st.info('this app builds a machine learning model!')

df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
