import streamlit as st

# Set page configuration
st.set_page_config(page_title="Dashboard", layout="wide")

# Create drop-down box with button options
selected_option = st.sidebar.selectbox("Select an option", 
                                       ("Home","Prediction", "Plots", "Wave Compare", "Brain Compare", "Stress Level"))

# Display corresponding content based on selected option
if selected_option == "Home":
    # Code for the "Prediction" button
    st.write("Home button selected.")
elif selected_option == "Prediction":
    # Code for the "Prediction" button
    st.write("Prediction button selected.")
elif selected_option == "Plots":
    # Code for the "Plots" button
    st.write("Plots button selected.")
elif selected_option == "Wave Compare":
    # Code for the "Wave Compare" button
    st.write("Wave Compare button selected.")
elif selected_option == "Brain Compare":
    # Code for the "Brain Compare" button
    st.write("Brain Compare button selected.")
elif selected_option == "Stress Level":
    # Code for the "Stress Level" button
    st.write("Stress Level button selected.")
