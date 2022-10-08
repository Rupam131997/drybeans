import streamlit as st
import pandas as pd
import numpy as np

st.title('Dry Beans Classification')
st.write('This is a simple classification app for dry beans dataset')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Area = st.sidebar.text_input("Area", 0)
    Perimeter = st.sidebar.text_input("Perimeter", 0)
    MajorAxisLength = st.sidebar.text_input("MajorAxisLength", 0)
    MinorAxisLength = st.sidebar.text_input("MinorAxisLength", 0)
    AspectRation = st.sidebar.text_input("AspectRation", 0)
    Eccentricity = st.sidebar.text_input("Eccentricity", 0)
    ConvexArea = st.sidebar.text_input("ConvexArea", 0)
    EquivDiameter = st.sidebar.text_input("EquivDiameter", 0)
    Extent = st.sidebar.text_input("Extent", 0)
    Solidity = st.sidebar.text_input("Solidity", 0)
    roundness = st.sidebar.text_input("roundness", 0)
    Compactness = st.sidebar.text_input("Compactness", 0)
    ShapeFactor1 = st.sidebar.text_input("ShapeFactor1", 0)
    ShapeFactor2 = st.sidebar.text_input("ShapeFactor2", 0)
    ShapeFactor3 = st.sidebar.text_input("ShapeFactor3", 0)
    ShapeFactor4 = st.sidebar.text_input("ShapeFactor4", 0)
    data = {'Area': Area,
            'Perimeter': Perimeter,
            'MajorAxisLength': MajorAxisLength,
            'MinorAxisLength': MinorAxisLength,
            'AspectRation': AspectRation,
            'Eccentricity': Eccentricity,
            'ConvexArea': ConvexArea,
            'EquivDiameter': EquivDiameter,
            'Extent': Extent,
            'Solidity': Solidity,
            'roundness': roundness,
            'Compactness': Compactness,
            'ShapeFactor1': ShapeFactor1,
            'ShapeFactor2': ShapeFactor2,
            'ShapeFactor3': ShapeFactor3,
            'ShapeFactor4': ShapeFactor4}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Load the data
df = pd.read_csv('Dry_Bean.csv')
# st.write(df)

# Convert the categorical data to numerical data
df['Class'] = df['Class'].astype('category')
df['Class'] = df['Class'].cat.codes

# Split the data into X and y
X = df.drop(['Class'], axis=1)
y = df['Class']

# Standardize the data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the model using Decision Tree Classifier
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# Build the model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Apply the model to make predictions
prediction = model.predict(input_df)

# Show the predictions with the corresponding class
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(model.predict_proba(input_df))

st.subheader('Accuracy Score')
st.write(model.score(X_test, y_test))

