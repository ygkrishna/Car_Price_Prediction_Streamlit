# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:39:50 2021

@author: xz328e
"""
#import os
#os.chdir('C:/Projects/GCP_ML/Car_Prediction_Model/streamlit/')

import pickle
import streamlit as st
import numpy as np
 
# loading the trained model
pickle_in = open('rf_reg.pkl', 'rb') 
model = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual, Year, Present_Price, Kms_Driven, Owner):   
    
    Fuel_Type_Diesel=0
    # Pre-processing user input    
    if(Fuel_Type_Petrol=='Petrol'):
        Fuel_Type_Petrol=1
        Fuel_Type_Diesel=0
    else:
        Fuel_Type_Petrol=0
        Fuel_Type_Diesel=1
 
    if(Seller_Type_Individual=='Individual'):
        Seller_Type_Individual=1
    else:
        Seller_Type_Individual=0
 
    if(Transmission_Mannual=='Mannual'):
        Transmission_Mannual=1
    else:
        Transmission_Mannual=0
 
    # Making predictions 
    prediction = model.predict(np.array([[Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Diesel, Fuel_Type_Petrol, 
                             Seller_Type_Individual, 
                             Transmission_Mannual]]))
    output=round(prediction[0],2)
     
    
    return output
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    #html_temp = """     """
      
    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Fuel_Type_Petrol = st.selectbox('Fuel_Type_Petrol',("Petrol","Diesel"))
    Seller_Type_Individual = st.selectbox('Seller_Type_Individual',("Individual","Dealer"))
    Transmission_Mannual = st.selectbox('Transmission_Mannual',("Manual Car","Automatic Car"))
    
    Year = st.number_input("Year of Manufacture") 
    Present_Price = st.number_input("Show room price")
    Kms_Driven = st.number_input("KM Driven Till Date")
    Owner = st.number_input("No of Previous Owners")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual, Year, Present_Price, Kms_Driven, Owner) 
        if result<0:
            st.success('Sorry you cannot sell this car {}'.format(result))
        else:
            st.success('You can sell the Car at {} lakhs'.format(result))
        #print(LoanAmount)
     
if __name__=='__main__': 
    main()