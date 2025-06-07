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

with st.expander('Data visualization'):
  chart = alt.Chart(df).mark_circle(size=60).encode(
    x='bill_length_mm',
    y='body_mass_g',
    color='species',
    tooltip=['species', 'bill_length_mm', 'body_mass_g']
).interactive()
  
  st.altair_chart(chart, use_container_width=True)
