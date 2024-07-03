import streamlit as st
import pickle
import numpy as np
import pandas as pd
model=pickle.load(open('model.pkl','rb'))

def diagnosis_prediction(input_data):
  
  a=np.asarray(input_data)
  b=a.reshape(1,-1)

  prediction=model.predict(b)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person is not diabetic'
  else:
    return 'The person is diabetic'

def main():
  st.title('Decision Tree Classification')
  Age=st.slider('Age',0,70)
  BMI=st.slider('BMI',0,50)
  Pregnancies=st.text_input('Pregnancies')
  Glucose=st.text_input('Glucose')
  BloodPressure=st.text_input('BloodPressure')
  SkinThickness=st.text_input('SkinThickness')
  Insulin=st.text_input('Insulin')
  DiabetesPedigreeFunction=st.text_input('DPF')
  df=pd.read_csv('/content/diabetes.csv')

  columns=['line','scatter','bar']
  for columns in st.multiselect('choose a plot',columns):
    if columns=='line':
      st.title('Lineplot')
      st.line_chart(df['Insulin'])    
    elif columns=='scatter':
      st.title('Scatterplot')
      st.scatter_chart(df['BloodPressure'])
    elif columns=='bar':
      st.title('Barplot')
      st.bar_chart(df['SkinThickness'])
  
  diagnosis=''


  if st.button('Diabetes Test Result'):
    diagnosis=diagnosis_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,DiabetesPedigreeFunction,Age,BMI])

  st.success(diagnosis)

if __name__ == '__main__':
  main()


st.markdown("""
<style>
.main {
  background-color: purple;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title('Profile')
st.sidebar.title('Settings')
st.sidebar.title('Help')
