import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and columns
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page config
st.set_page_config(
    page_title="Autovaluer - Car Price Predictor 🚗", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header with emoji and colors
st.title("🚗 Autovaluer -  Car Price Predictor")
st.markdown("##### Get instant price estimates for your dream car")
st.divider()

# Info banner
st.info("💡 Fill in the details below to get an accurate price prediction")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox(
        "🏭 Manufacturer", 
        ["Ford", "Toyota", "Porsche", "VW", "BMW"],
        help="Select the car manufacturer"
    )
    
    fuel_type = st.selectbox(
        "⛽ Fuel Type", 
        ["Petrol", "Diesel", "Hybrid"],
        help="Choose the fuel type"
    )
    
    year = st.number_input(
        "📅 Year of Manufacture", 
        min_value=1990, 
        max_value=2025, 
        value=2020,
        step=1,
        help="Enter the manufacturing year"
    )

with col2:
    model_name = st.selectbox(
        "🚘 Model", 
        ["Fiesta", "Focus", "911", "Cayenne", "Golf", "Mondeo", 
         "Polo", "Passet", "RAV4", "Prius", "Yaris", "Z4", 
         "M5", "X3", "718 Cayman"],
        help="Select the car model"
    )
    
    engine_size = st.number_input(
        "🔧 Engine Size (L)", 
        min_value=1.0, 
        max_value=8.0,
        value=2.0,
        step=0.1,
        help="Enter engine size in liters"
    )
    
    mileage = st.number_input(
        "📏 Mileage (km)", 
        min_value=0, 
        max_value=500000,
        value=50000,
        step=1000,
        help="Enter total mileage"
    )

st.divider()

# Predict button
if st.button("🔮 Predict Price Now", type="secondary", width="stretch"):
    with st.spinner("🔄 Calculating price..."):
        try:
            input_df = pd.DataFrame([{
                'Manufacturer': manufacturer,
                'Model': model_name,
                'Fuel type': fuel_type,
                'Engine size': engine_size,
                'Year of manufacture': year,
                'Mileage': mileage
            }])

            input_encoded = pd.get_dummies(input_df, drop_first=True)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            pred = model.predict(input_encoded)
            pred_exp = np.exp(pred)
            predicted_price = round(pred_exp[0], 2)

            # Display result
            st.balloons()
            st.success(f"### 💰 Estimated Price: ₹ {predicted_price:,.2f}")
            
            # Show car details in expandable section
            with st.expander("📋 View Full Details"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.metric("Manufacturer", manufacturer)
                    st.metric("Model", model_name)
                    st.metric("Fuel Type", fuel_type)
                
                with detail_col2:
                    st.metric("Year", year)
                    st.metric("Engine Size", f"{engine_size:,.2f} L")
                    st.metric("Mileage", f"{mileage:,} km")
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.divider()
st.caption("💡 Tip: Try different combinations to compare prices across various models")
st.caption("🔒 Your data is processed securely and not stored")