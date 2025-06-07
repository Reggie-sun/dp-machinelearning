import streamlit as st
import pandas as pd

st.title('machine learning app')

st.info('this app builds a machine learning model!')
with st.expander("Data"):
  st.write("**Raw data**")
  df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df
x=df.drop("species",axis=1)
y=df['species']

with st.expander('Data visualization')
  st.st.scatter_chart(df,x='bill_length_mm',y='body_mass_g',color='species')
