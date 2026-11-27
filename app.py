import streamlit as st, requests
st.title("Iris classifier")
sl = st.slider("Sepal length", 4.0, 8.0, 5.8)
sw = st.slider("Sepal width", 2.0, 4.5, 3.0)
pl = st.slider("Petal length", 1.0, 7.0, 4.3)
pw = st.slider("Petal width", 0.1, 2.5, 1.3)

CLASSES = ["setosa", "versicolor", "virginica"]

if st.button("Predict"):
    r = requests.post("http://localhost:8000/predict",
                      json={"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw})
    pred = r.json()['prediction']
    st.success(f"Predicted Class: {CLASSES[pred]}")