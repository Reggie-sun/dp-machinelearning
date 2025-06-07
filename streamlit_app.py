import streamlit as st
import pandas as pd
import altair as alt

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

def slider():

  slider_lables=[("Island", "island"),
("Bill Length (mm)", "bill_length_mm"),
("Bill Depth (mm)", "bill_depth_mm"),
("Flipper Length (mm)", "flipper_length_mm"),
("Body Mass (g)", "body_mass_g"),
("Sex", "sex")]

inout_data={}

  for lables,key in slider_lables:
    input_data[key]=st.slider(laber,min_value=float(0), max_value=float(df[key].max()), value=float(df[key].mean()))
  
  




