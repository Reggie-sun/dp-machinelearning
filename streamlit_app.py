import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title('machine learning app')

st.info('this app builds a machine learning model!')
with st.expander("Data"):
    st.write("**Raw data**")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
    df

x_raw = df.drop("species", axis=1)
y_raw = df['species']
x_slider = x_raw.drop(["island", "sex"], axis=1)

with st.expander('Data visualization'):
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_length_mm',
        y='body_mass_g',
        color='species',
        tooltip=['species', 'bill_length_mm', 'body_mass_g']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, x_raw], axis=0)


with st.expander("input features"):
    st.write("**input penguin**")
    input_df  # 没有进行单热编码
    st.write("**combined penguins data**")
    input_penguins

# 对输入的x数据进行单热编码(不是传统的0，1这些，而是通过打勾的方式)
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
x_raw = df_penguins[1:]  # 取下标为第一排开始的所有数据，就是除了输入数据的所有数据，因为输入数据在第一排（下表为0）
input_row = df_penguins[:1]  # 取下标为0的数据，即真正的第一排，自己通过构建滑块输入的数据

# 对y进行单热编码
y_raw = y_raw.map({'Adelie': 0,
                   'Chinstrap': 1,
                   'Gentoo': 2})


with st.expander('data preparation'):
    st.write('**encoded input x**')
    input_row  # 进行了单热编码
    st.write('**encoded y**')
    y_raw

model = RandomForestClassifier()
model.fit(x_raw, y_raw)
y_pred = prediction = model.predict(input_row)
predict_proba = model.predict_proba(input_row)
predict_proba = pd.DataFrame(predict_proba)
predict_proba.rename(columns={
    0: 'Adelie',
    1: 'Chinstrap',
    2: 'Gentoo'
}, inplace=True)
predict_proba
st.dataframe(predict_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn(
                     'Adelie',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Chinstrap': st.column_config.ProgressColumn(
                     'Chinstrap',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Gentoo': st.column_config.ProgressColumn(
                     'Gentoo',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
             }, hide_index=True)
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
  
  




