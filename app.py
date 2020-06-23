"""
Enter Sepal.Length , Sepal.Width,Petal.Length,Petal.Width

and know about Species  = setosa,versicolor,virginica
"""

import numpy as np
import pickle
import pandas as pd

import streamlit as st 

from PIL import Image

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)
def predict_species(SL,SW,PL,PW):
    prediction=classifier.predict([[SL,SW,PL,PW]])
    print(prediction)
    return prediction
    





def main():
    st.title("Know about Species")
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Species ML App</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)

    SL=st.text_input("Sepal.Length","Type Sepal Length")
    
    SW=st.text_input("Sepal.Width","Type Sepal Width")
    
    PL=st.text_input("Petal.Length","Type Petal Length")

    PW=st.text_input("Petal.Width","Type Petal Width")

    result=""

    if st.button("Predict"):
        result=predict_species(SL,SW,PL,PW)

    st.success("The output is {}".format(result))

    if st.button("About"):
        st.text("Simple StreamLit Implementation")
        st.text("Thanks to Krish Naik")


if __name__ =='__main__':
    main()

