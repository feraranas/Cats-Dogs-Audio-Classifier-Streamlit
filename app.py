import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Hello world!")

df = pd.read_csv('titanic.csv')
st.write(df)
# Add some matplotlib code !
fig, ax = plt.subplots()
df.hist(
  bins=8,
  column="Age",
  grid=False,
  figsize=(8, 8),
  color="#86bf91",
  zorder=2,
  rwidth=0.9,
  ax=ax,
)
st.write(fig)