import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jmvazqueznicolas/AI_and_DS_Tec2023/main/Activity%203%20ETL%20code/pima-indians-diabetes.data.csv'
df = pd.read_csv(url)

st.title('Actividad 3: Indian Diabetes')
st.header('Equipo 4')
team = pd.DataFrame({
     'Alumno': [
         'Mauricio Juárez Sánchez',
         'Alfredo Jeong Hyun Park',
         'Fernando Alfonso Arana Salas',
         'Miguel Ángel Bustamante Pérez'],
     'Matricula': [
      'A01660336',
      'A01658259',
      'A01272933',
      'A01781583']
     })
st.write(team)

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
diabetes = pd.DataFrame(df.values, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"],  index=pd.RangeIndex(df.index))
st.write('Original dataset')
st.write(diabetes)

st.text('1. Transformation')
st.caption('Apply data imputation for the columns: Glucose, BloodPressure, SkinThickness, Insulin, and BMI. In this dataset, a zero value is considered a missing value.')
transformation = '''
     # Imputation with SimpleImputer from Sklearn
     from sklearn.impute import SimpleImputer

     # Replace values with numpy
     import numpy as np

     # Apply data imputation for this columns:
     columns_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

     # First we convert zero values to numpy.nan
     diabetes[columns_to_impute] = diabetes[columns_to_impute].replace(0, np.nan)

     # Select strategy 'median' of each feature
     imputer = SimpleImputer(strategy="median")

     # Train the imputer and transform the database
     diabetes[columns_to_impute] = imputer.fit_transform(diabetes[columns_to_impute])
     '''
st.code(transformation, language="python", line_numbers=False)

columns_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
diabetes[columns_to_impute] = diabetes[columns_to_impute].replace(0, np.nan)
imputer = SimpleImputer(strategy="median")
diabetes[columns_to_impute] = imputer.fit_transform(diabetes[columns_to_impute])

st.write('Imputed dataset')
st.write(diabetes)

st.text('2. Feature Engineering')
st.caption('Apply feature engineering to create a new data column to categorize the age into three categories.')
featureEngineering = '''
     # Select Ages
     ages = diabetes["Age"].tolist()

     # Defining the bin edges for different age groups
     bin_edges = [0, 29, 55, 100]  # Age ranges: 0-29, 30-55, 56-100

     # Using pd.cut() to categorize ages into bins, into new feature
     diabetes["ages_cat"] = pd.cut(ages,
                                   bins=bin_edges,
                                   labels=["0-29", "30-55", "56+"])
'''
st.code(featureEngineering, language="python", line_numbers=False)
ages = diabetes["Age"].tolist()
bin_edges = [0, 29, 55, diabetes["Age"].max()]
diabetes["ages_cat"] = pd.cut(ages,
                              bins=bin_edges,
                              labels=["Young", "Young-Adult", "Adult"])


st.write('Showing distribution of new feature grouped')
fig, ax = plt.subplots()
diabetes["ages_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True, ax=ax)
plt.xlabel("Ages category")
plt.ylabel("Number of individuals")
st.pyplot(fig)

st.write('New category dataset')
st.write(diabetes)

st.text('3. Load')
st.caption('Load the modified dataset in a CSV and Json format.')
loadDataset = '''
     # Load modified dataset to CSV
     filename_csv = "modified_diabetes.csv"
     diabetes.to_csv(filename_csv, index=False)

     # Load modified dataset to JSON
     filename_json = "modified_diabetes.json"
     diabetes.to_json(filename_json, orient="records", lines=True)

     print("Modified dataset saved to CSV:", filename_csv)
     print("Modified diabetes saved to JSON:", filename_json)
'''
st.code(loadDataset, language="python", line_numbers=False)