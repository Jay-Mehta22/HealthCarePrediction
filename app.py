import base64
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF, XPos, YPos
from database import *

# ---------------- PAGE CONFIG (must be first) ----------------
st.set_page_config(page_title="Healthcare AI", layout="centered")

# ---------------- INIT ----------------
create_tables()

if "user" not in st.session_state:
    st.session_state.user = None

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# ---------------- LOGIN ----------------
if st.session_state.user is None:
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Signup"):
            if create_user(nu, np):
                st.success("Account created")
            else:
                st.error("User exists")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Healthcare AI")
st.sidebar.success(f"User: {st.session_state.user}")
st.sidebar.warning("⚠️ Educational use only")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.last_pred = None
    st.rerun()

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model    = joblib.load("heart_model.pkl")
heart_scaler   = joblib.load("heart_scaler.pkl")

st.title("🩺 Healthcare AI Predictor")

# ---------------- TABS ----------------
t1, t2, t3 = st.tabs(["🩸 Diabetes", "❤️ Heart", "📊 History"])

# ---------------- PDF ----------------
def generate_pdf(user, disease, result, prob, inputs):
    pdf = FPDF()
    pdf.add_page()

    # FIX: use Helvetica instead of Arial (Arial deprecated in fpdf2)
    # FIX: use new_x/new_y instead of ln=True (ln deprecated in fpdf2)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        200, 12, "Healthcare AI - Medical Report",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"
    )
    pdf.ln(3)

    # Patient & prediction section
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(
        200, 8, "Patient & Prediction Details",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(200, 7, f"Username   : {user}",              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 7, f"Disease    : {disease}",           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 7, f"Result     : {result}",            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 7, f"Probability: {prob * 100:.2f}%",   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    risk_level = (
        "Critical" if prob * 100 > 75
        else "Moderate" if prob * 100 > 40
        else "Normal"
    )
    pdf.cell(200, 7, f"Risk Level : {risk_level}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # All input parameters section
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(
        200, 8, "All Input Parameters",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    for k, v in inputs.items():
        pdf.cell(200, 7, f"  {k}: {v}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Disclaimer
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(200, 7, "Disclaimer: This report is for educational purposes only.",        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 7, "Please consult a qualified healthcare professional for diagnosis.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    file = "report.pdf"
    pdf.output(file)
    return file

# ---------------- RESULT ----------------
def show_result(pred, prob, disease, inputs):

    result = "High Risk" if pred == 1 else "Low Risk"

    # Prevent duplicate history saves on Streamlit reruns
    pred_key = (disease, result, float(prob))
    if st.session_state.last_pred != pred_key:
        save_history(st.session_state.user, disease, result, float(prob))
        st.session_state.last_pred = pred_key

    st.markdown("---")
    risk_percent = prob * 100

    # ================= RESULT CARD =================
    if pred == 1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#7f1d1d,#450a0a);
            padding:25px; border-radius:20px;
            border:1px solid rgba(255,0,0,0.3);
            box-shadow:0 0 25px rgba(255,0,0,0.2);
        ">
            <h2 style="color:white;">🚨 High Risk Detected</h2>
            <h1 style="color:#f87171;">{risk_percent:.2f}%</h1>
            <p style="color:#fecaca;">Immediate lifestyle monitoring recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#064e3b,#022c22);
            padding:25px; border-radius:20px;
            border:1px solid rgba(0,255,100,0.2);
            box-shadow:0 0 25px rgba(0,255,100,0.15);
        ">
            <h2 style="color:white;">✅ Low Risk</h2>
            <h1 style="color:#4ade80;">{100 - risk_percent:.2f}% Safe</h1>
            <p style="color:#bbf7d0;">Maintain healthy habits and regular monitoring.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= KPI CARDS =================
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk %", f"{risk_percent:.1f}%")
    c2.metric("Status", result)
    level = (
        "Critical" if risk_percent > 75
        else "Moderate" if risk_percent > 40
        else "Normal"
    )
    c3.metric("Risk Level", level)

    # ================= GAUGE CHART =================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': f"{disease} Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ec4899"},
            'steps': [
                {'range': [0, 35],   'color': "#14532d"},
                {'range': [35, 70],  'color': "#854d0e"},
                {'range': [70, 100], 'color': "#7f1d1d"}
            ],
        }
    ))
    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    # FIX: use_container_width deprecated — replaced with width='stretch'
    st.plotly_chart(fig, width='stretch')

    # ================= INPUT SUMMARY TABLE =================
    st.markdown("## 📋 Input Summary")
    summary_df = pd.DataFrame(list(inputs.items()), columns=["Parameter", "Value"])
    summary_df["Value"] = summary_df["Value"].astype(str)
    # FIX: use_container_width deprecated — replaced with width='stretch'
    st.dataframe(summary_df, width='stretch', hide_index=True)

    # ================= SMART RECOMMENDATIONS =================
    st.markdown("## 🧠 Smart Recommendations")

    if disease == "Diabetes":
        if risk_percent > 70:
            st.warning("""
            • Reduce sugar intake immediately
            • Exercise daily
            • Monitor glucose regularly
            • Consult a diabetologist
            """)
        elif risk_percent > 40:
            st.info("""
            • Maintain balanced diet
            • Reduce processed foods
            • Track glucose weekly
            """)
        else:
            st.success("""
            • Maintain healthy lifestyle
            • Continue regular exercise
            """)

    elif disease == "Heart":
        if risk_percent > 70:
            st.warning("""
            • Reduce cholesterol intake
            • Avoid smoking/alcohol
            • Monitor BP regularly
            • Immediate cardiac consultation advised
            """)
        elif risk_percent > 40:
            st.info("""
            • Maintain cardio exercise
            • Reduce sodium intake
            • Sleep properly
            """)
        else:
            st.success("""
            • Heart condition appears stable
            • Continue healthy routine
            """)

    # ================= PDF REPORT =================
    st.markdown("## 📄 Medical Report")

    file = generate_pdf(st.session_state.user, disease, result, prob, inputs)

    with open(file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label="📥 Download PDF Report",
        data=PDFbyte,
        file_name="Healthcare_Report.pdf",
        mime="application/pdf"
    )

# ================= DIABETES TAB =================
with t1:
    st.subheader("Diabetes Prediction")

    c1, c2 = st.columns(2)

    with c1:
        preg    = st.number_input("Pregnancies", 0, 20, key="d_preg")
        glucose = st.number_input("Glucose", 0, 300, key="d_glu")
        bp      = st.number_input("Blood Pressure", 0, 200, key="d_bp")
        skin    = st.number_input("Skin Thickness", 0, 100, key="d_skin")

    with c2:
        insulin = st.number_input("Insulin", 0, 900, key="d_ins")
        bmi     = st.number_input("BMI", 0.0, 70.0, key="d_bmi")
        dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, key="d_dpf")
        age     = st.number_input("Age", 1, 120, key="d_age")

    if st.button("Predict Diabetes"):
        if glucose <= 0 or bmi <= 0:
            st.error("❌ Invalid values. Glucose and BMI must be greater than 0.")
        else:
            df_input = pd.DataFrame([{
                "Pregnancies":              preg,
                "Glucose":                  glucose,
                "BloodPressure":            bp,
                "SkinThickness":            skin,
                "Insulin":                  insulin,
                "BMI":                      bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age":                      age
            }])

            inputs = {
                "Pregnancies":                preg,
                "Glucose (mg/dL)":            glucose,
                "Blood Pressure (mmHg)":      bp,
                "Skin Thickness (mm)":        skin,
                "Insulin (mu U/ml)":          insulin,
                "BMI":                        round(float(bmi), 1),
                "Diabetes Pedigree Function": round(float(dpf), 3),
                "Age":                        age
            }

            pred = diabetes_model.predict(df_input)[0]
            prob = diabetes_model.predict_proba(df_input)[0][1]

            show_result(pred, prob, "Diabetes", inputs)

# ================= HEART TAB =================
with t2:
    st.subheader("Heart Disease Prediction")

    c1, c2 = st.columns(2)

    with c1:
        age_h    = st.number_input("Age", 1, 120, key="h_age")
        sex      = st.selectbox("Sex (0=Female, 1=Male)", [0, 1], key="h_sex")
        cp       = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], key="h_cp")
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, key="h_bp")
        chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, key="h_chol")
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL (0=No, 1=Yes)", [0, 1], key="h_fbs")

    with c2:
        restecg  = st.selectbox("Resting ECG (0-2)", [0, 1, 2], key="h_ecg")
        thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, key="h_hr")
        exang    = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1], key="h_ex")
        oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, key="h_old")
        slope    = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2], key="h_slope")
        ca       = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3], key="h_ca")
        thal     = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3], key="h_thal")

    if st.button("Predict Heart"):
        if chol <= 0 or trestbps <= 0 or thalach <= 0 or oldpeak < 0:
            st.error("❌ Please enter valid medical values")

        elif age_h < 18:
            st.error("❌ Age should be greater than 18")

        elif chol < 100 or chol > 600:
            st.error("❌ Cholesterol value looks abnormal")

        elif trestbps < 80 or trestbps > 220:
            st.error("❌ Blood pressure value invalid")

        elif thalach < 60 or thalach > 220:
            st.error("❌ Max heart rate value invalid")

        else:
            df_input = pd.DataFrame([{
                "age":      age_h,
                "sex":      sex,
                "cp":       cp,
                "trestbps": trestbps,
                "chol":     chol,
                "fbs":      fbs,
                "restecg":  restecg,
                "thalach":  thalach,
                "exang":    exang,
                "oldpeak":  oldpeak,
                "slope":    slope,
                "ca":       ca,
                "thal":     thal
            }])

            inputs = {
                "Age":                           age_h,
                "Sex":                           "Male" if sex == 1 else "Female",
                "Chest Pain Type":               cp,
                "Resting Blood Pressure (mmHg)": trestbps,
                "Cholesterol (mg/dL)":           chol,
                "Fasting Blood Sugar > 120":     "Yes" if fbs == 1 else "No",
                "Resting ECG":                   restecg,
                "Max Heart Rate Achieved":       thalach,
                "Exercise Induced Angina":       "Yes" if exang == 1 else "No",
                "ST Depression (Oldpeak)":       round(float(oldpeak), 1),
                "Slope of ST Segment":           slope,
                "Major Vessels Colored":         ca,
                "Thalassemia":                   thal
            }

            scaled_data = heart_scaler.transform(df_input)

            pred = heart_model.predict(scaled_data)[0]
            prob = heart_model.predict_proba(scaled_data)[0][1]

            show_result(pred, prob, "Heart", inputs)

# ================= HISTORY TAB =================
with t3:
    st.subheader("📊 History")

    data = get_history(st.session_state.user)

    if data:
        df = pd.DataFrame(data, columns=["Disease", "Result", "Probability", "Time"])
        df["Probability"] *= 100
        df["Date"] = pd.to_datetime(df["Time"]).dt.strftime("%d %b")

        # Diabetes trend
        st.markdown("### 🩸 Diabetes Trend")
        d = df[df["Disease"] == "Diabetes"]

        if not d.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=d["Date"], y=d["Probability"],
                mode="lines+markers",
                line=dict(width=3, shape="spline"),
                marker=dict(
                    size=10,
                    color=d["Result"].map({"High Risk": "red", "Low Risk": "green"})
                )
            ))
            fig.update_layout(yaxis=dict(range=[0, 100]), template="plotly_dark")
            # FIX: use_container_width deprecated — replaced with width='stretch'
            st.plotly_chart(fig, width='stretch')
            latest = d.iloc[-1]
            st.caption(f"Latest: {latest['Probability']:.1f}% ({latest['Result']})")
        else:
            st.info("No diabetes history")

        # Heart trend
        st.markdown("### ❤️ Heart Trend")
        h = df[df["Disease"] == "Heart"]

        if not h.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=h["Date"], y=h["Probability"],
                mode="lines+markers",
                line=dict(width=3, shape="spline"),
                marker=dict(
                    size=10,
                    color=h["Result"].map({"High Risk": "red", "Low Risk": "green"})
                )
            ))
            fig.update_layout(yaxis=dict(range=[0, 100]), template="plotly_dark")
            # FIX: use_container_width deprecated — replaced with width='stretch'
            st.plotly_chart(fig, width='stretch')
            latest = h.iloc[-1]
            st.caption(f"Latest: {latest['Probability']:.1f}% ({latest['Result']})")
        else:
            st.info("No heart history")

    else:
        st.info("No history yet")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Final Year Project | Industry Ready")