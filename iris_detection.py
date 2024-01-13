import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

st.write(
    """
# Simple Iris Flower prediction App
         
This app predicts the Iris flower type!
"""
)

st.sidebar.header("User Input Parameters")


def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    option = st.sidebar.selectbox(
        "Which machine learning model would you like to use?",
        ("Random Forest", "SVM", "Logistic Regression"),
    )

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    features = pd.DataFrame(data, index=[0])

    return features, option


df = user_input_features()

st.subheader("User Input parameters")
st.write(df[0])

iris = datasets.load_iris()
X = iris.data
y = iris.target

if df[1] == "Random Forest":
    clf = RandomForestClassifier()
elif df[1] == "SVM":
    clf = SVC(probability=True)  # Enable probability estimates
elif df[1] == "Logistic Regression":
    clf = LogisticRegression()

clf.fit(X, y)

prediction = clf.predict(df[0])
prediction_proba = clf.predict_proba(df[0])

st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

st.subheader("Prediction")
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)
