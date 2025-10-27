import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Test Dashboard",
    page_icon="🎮",
    layout="wide"
)

# Simple test page
st.title("🎮 StormGate Test Dashboard")
st.write("This is a test to verify Streamlit is working correctly.")

# Create some test data
test_data = {
    'Aspect': ['General', 'Gameplay', 'Graphics', 'Audio', 'Controls'],
    'Reviews': [612, 371, 250, 180, 150],
    'Positive Rate': [0.45, 0.52, 0.38, 0.41, 0.48]
}

df = pd.DataFrame(test_data)

# Display data table
st.subheader("📊 Test Data")
st.dataframe(df, use_container_width=True)

# Display simple chart
st.subheader("📈 Test Chart")
st.bar_chart(df.set_index('Aspect')['Reviews'])

st.success("✅ Streamlit is working correctly!")
st.info("If you can see this page, your dashboard setup is ready!")
