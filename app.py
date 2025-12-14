import pandas as pd
import pickle as pk
import streamlit as st

data = pd.read_csv("cleaned_hyd_housing_data_final1.csv")
model = pk.load(open("hyd_house_pred_model.pkl", 'rb'))

st.set_page_config(page_title="Price Predictor", layout="centered",page_icon="🏠")
st.header("🏠Hyderabad House Price Predictor")

left,mid, right = st.columns([4,.5,3])

with left:
    loc = st.selectbox('Choose location', data['locality'].unique())
    sqft = st.number_input("Enter total sqft",min_value=1000,step=50)
    beds = st.number_input("Enter no.of bedrooms",min_value=1,value=2,step=1)
    baths = st.number_input("Enter no.of bathrooms",min_value=1,value=2,step=1)
    floor = st.number_input("Enter floor number",min_value=0,value=2,step=1)
    type = st.selectbox("Choose property type", data['property_type'].unique())
    furnish = st.selectbox("Choose furnished status", data['furnished_status'].unique())
#isnew = st.selectbox("Choose (1)New/(0)Resale",[0,1])

input_df = pd.DataFrame([[loc, sqft, beds, baths, floor, type, furnish, 1]],
                        columns=['locality', 'sqft', 'bedrooms2', 'bathrooms', 'floor', 'property_type', 'furnished_status', 'brand_new'])
with right:
    st.markdown("<br>" * 7, unsafe_allow_html=True)
    if st.button("Predict House Price",width="content"):
        output = model.predict(input_df)
        st.success(f"Predicted house price is: ₹{output[0]:,.2f}",width="stretch")
