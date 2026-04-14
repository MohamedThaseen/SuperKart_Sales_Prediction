


import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("Product_Store_Sales_model_v1_0.joblib")

model = load_model()

st.title("Product Sales Prediction App")
st.subheader("Enter Product & Store Details")


Product_Weight = st.number_input("Product_Weight (Min: 4.0, Max: 22.0)*", min_value=6.60, max_value=19.0)
Product_Sugar_Content = st.selectbox("Sugar_Content*", ['Low Sugar','Regular','No Sugar','reg'])
Product_Allocated_Area = st.number_input("Product_Allocated_Area (Min: 0.4, Max: 29.8)*", min_value=0.4, max_value=29.8)
Product_Type = st.selectbox("Product_Type *", ['Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Snack Foods',
                                            'Meat','Household','Hard Drinks','Fruits and Vegetables',
                                            'Breads','Soft Drinks','Breakfast','Others','Starchy Foods','Seafood'])
Product_MRP = st.number_input("Product_Price (Min: 31.0, Max: 266.0)*", min_value=31.0, max_value=266.0)
Store_Id = st.selectbox("Store_Id (Optional)", ['OUT004','OUT003','OUT001','OUT002'])
Store_Establishment_Year = st.selectbox("Store_Establishment_Year (Optional)", ['2009','1999','1987','1998'])
Store_Size = st.selectbox("Store_Size (Optional)", ['Medium','High','Small'])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type (Optional)", ['Tier 1','Tier 2','Tier 3'])
Store_Type = st.selectbox("Store_Type *", ['Supermarket Type1','Supermarket Type2','Departmental Store','Food Mart'])


input_data = pd.DataFrame([{

        'Product_Weight': float(Product_Weight),
        'Product_Sugar_Content': Product_Sugar_Content,
        'Product_Allocated_Area': float(Product_Allocated_Area),
        'Product_Type': Product_Type,
        'Product_MRP': float(Product_MRP),
        'Store_Id': Store_Id,
        'Store_Establishment_Year': int(Store_Establishment_Year),
        'Store_Size': Store_Size,
        'Store_Location_City_Type': Store_Location_City_Type,
        'Store_Type': Store_Type
    }])

if st.button("🔮 Predict Sales"):
  Prediction = model.predict(input_data)[0]
  st.info(f"Predicted Sales:{Prediction:.2f} ")
