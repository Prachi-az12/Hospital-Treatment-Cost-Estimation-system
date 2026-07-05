import streamlit as st
import pandas as pd
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Hospital Treatment Cost Estimator",
    page_icon="🏥",
    layout="wide"
)

# ------------------ Load Model ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ CSS ------------------
st.markdown("""
<style>

.main{
    background-color:#f4f8fb;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0b5ed7;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    background:linear-gradient(to right,#0d6efd,#00c6ff);
    color:white;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:linear-gradient(to right,#0056b3,#0096c7);
}

.result{
    background:#d1e7dd;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:#0f5132;
}

.box{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown('<p class="title">🏥 Hospital Treatment Cost Estimator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the estimated medical treatment cost based on patient details.</p>', unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.title("ℹ Information")
st.sidebar.write("""
This application predicts the estimated treatment cost using Machine Learning.

Fill all patient details and click **Predict Cost**.
""")

# ------------------ Form ------------------
st.markdown('<div class="box">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 18, 100, 30)
    bmi = st.number_input("⚖ BMI", 10.0, 60.0, 25.0)
    smoker = st.selectbox("🚬 Smoker", ["yes", "no"])

with col2:
    sex = st.selectbox("🧑 Gender", ["male", "female"])
    children = st.number_input("👶 Children", 0, 5, 0)
    region = st.selectbox(
        "🌍 Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Data ------------------
input_data = pd.DataFrame({
    "age":[age],
    "sex":[sex],
    "bmi":[bmi],
    "children":[children],
    "smoker":[smoker],
    "region":[region]
})

input_data = pd.get_dummies(input_data, drop_first=True)

missing_cols = set(model.feature_names_in_) - set(input_data.columns)

for col in missing_cols:
    input_data[col] = 0

input_data = input_data[model.feature_names_in_]

# ------------------ Prediction ------------------
if st.button("💰 Predict Treatment Cost"):

    prediction = model.predict(input_data)[0]

    st.balloons()

    st.markdown(
        f"""
        <div class="result">
        💰 Estimated Treatment Cost <br><br>
        ₹ {prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
