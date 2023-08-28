import os

# Create the .streamlit directory if it doesn't exist
streamlit_dir = os.path.expanduser("~/.streamlit")
if not os.path.exists(streamlit_dir):
    os.makedirs(streamlit_dir)

# Write credentials.toml
credentials_path = os.path.join(streamlit_dir, "credentials.toml")
with open(credentials_path, "w") as f:
    f.write("[general]\n")
    f.write("email = \"feraranas@gmail.com\"\n")

# Write config.toml
config_path = os.path.join(streamlit_dir, "config.toml")
with open(config_path, "w") as f:
    f.write("[server]\n")
    f.write("headless = true\n")
    f.write("enableCORS = false\n")
    f.write("port = $PORT\n")

import streamlit as st

st.title('Greatest of Three Numbers') # Set Title of the webapp

choice1 = st.number_input('Enter First number') #Accepts a number input
choice2 = st.number_input('Enter Second number')
choice3 = st.number_input('Enter Third number')

string = f'Maximum value is {max(choice1,choice2,choice3)}'

st.write(string)