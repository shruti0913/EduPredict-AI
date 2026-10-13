
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# 1. APP CONFIGURATION
# -------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_risk_pipeline.joblib"
DATA_PATH = ROOT / "data" / "student-mat.csv"

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# 2. LOAD MODEL AND DATA
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

try:
    model = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

# -------------------------------
# 3. SIDEBAR
# -------------------------------
st.sidebar.title("🎓 EduPredict AI")
st.sidebar.write("Student Academic Risk Prediction")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Risk Prediction", "About Project"]
)

st.sidebar.divider()
st.sidebar.caption("Educational prototype | ML + Explainable AI")

# -------------------------------
# 4. DASHBOARD
# -------------------------------
if page == "Dashboard":

    st.title("🎓 EduPredict AI")
    st.subheader("Student Academic Risk Analytics")

    st.write(
        "Explore student performance patterns and estimate "
        "academic risk using a machine learning model."
    )

    total_students = len(df)
    at_risk_count = int((df["G3"] < 10).sum())
    not_at_risk_count = total_students - at_risk_count

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Students", total_students)
    c2.metric("At-Risk Students", at_risk_count)
    c3.metric("Not-at-Risk Students", not_at_risk_count)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📊 Academic Risk Distribution")

        risk_chart = pd.DataFrame({
            "Category": ["At Risk", "Not At Risk"],
            "Students": [at_risk_count, not_at_risk_count]
        }).set_index("Category")

        st.bar_chart(risk_chart)

    with right:
        st.subheader("📈 Final Grade Distribution")

        grade_chart = (
            df["G3"].value_counts()
            .sort_index()
            .rename_axis("Final Grade")
            .to_frame("Students")
        )

        st.bar_chart(grade_chart)

    st.divider()

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "Dataset: UCI Student Performance. "
        "At-risk label is defined as G3 < 10."
    )

# -------------------------------
# 5. RISK PREDICTION
# -------------------------------
elif page == "Risk Prediction":

    st.title("🔍 Student Risk Prediction")

    st.write(
        "Enter student details to estimate academic risk. "
        "Grades G1, G2 and G3 are excluded from model inputs."
    )

    with st.form("student_form"):

        st.subheader("Personal and Family Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U", "R"])
            famsize = st.selectbox("Family Size", ["GT3", "LE3"])
            Pstatus = st.selectbox(
                "Parents' Cohabitation", ["T", "A"]
            )

        with col2:
            Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
            Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])
            Mjob = st.selectbox(
                "Mother's Job",
                ["teacher", "health", "services", "at_home", "other"]
            )
            Fjob = st.selectbox(
                "Father's Job",
                ["teacher", "health", "services", "at_home", "other"]
            )
            reason = st.selectbox(
                "School Choice Reason",
                ["home", "reputation", "course", "other"]
            )
            guardian = st.selectbox(
                "Guardian", ["mother", "father", "other"]
            )

        st.divider()
        st.subheader("Study Habits and Academic Support")

        col4, col5, col6 = st.columns(3)

        with col4:
            traveltime = st.selectbox(
                "Travel Time (1–4)", [1, 2, 3, 4]
            )
            studytime = st.selectbox(
                "Weekly Study Time (1–4)", [1, 2, 3, 4]
            )
            failures = st.selectbox(
                "Past Class Failures", [0, 1, 2, 3]
            )
            schoolsup = st.selectbox(
                "School Support", ["yes", "no"]
            )
            famsup = st.selectbox(
                "Family Support", ["yes", "no"]
            )
            paid = st.selectbox(
                "Extra Paid Classes", ["yes", "no"]
            )

        with col5:
            activities = st.selectbox(
                "Extracurricular Activities", ["yes", "no"]
            )
            nursery = st.selectbox(
                "Attended Nursery School", ["yes", "no"]
            )
            higher = st.selectbox(
                "Wants Higher Education", ["yes", "no"]
            )
            internet = st.selectbox(
                "Internet Access", ["yes", "no"]
            )
            romantic = st.selectbox(
                "Romantic Relationship", ["yes", "no"]
            )

        with col6:
            famrel = st.slider(
                "Family Relationship (1–5)", 1, 5, 4
            )
            freetime = st.slider(
                "Free Time (1–5)", 1, 5, 3
            )
            goout = st.slider(
                "Going Out (1–5)", 1, 5, 3
            )
            Dalc = st.slider(
                "Workday Alcohol Scale (1–5)", 1, 5, 1
            )
            Walc = st.slider(
                "Weekend Alcohol Scale (1–5)", 1, 5, 1
            )

        st.divider()
        st.subheader("Health and Attendance")

        col7, col8 = st.columns(2)

        with col7:
            health = st.slider("Health (1–5)", 1, 5, 3)

        with col8:
            absences = st.number_input(
                "School Absences",
                min_value=0,
                max_value=100,
                value=0
            )

        submitted = st.form_submit_button(
            "Predict Academic Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:

        student = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences
        }

        student_df = pd.DataFrame([student])

        prediction = int(model.predict(student_df)[0])

        st.divider()
        st.subheader("📌 Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(student_df)[0]
            classes = list(model.named_steps["classifier"].classes_)
            risk_index = classes.index(1)
            risk_probability = float(probabilities[risk_index])
        else:
            risk_probability = None

        if prediction == 1:
            st.error("⚠️ The model estimates elevated academic risk.")
            st.write(
                "Consider supportive check-ins, tutoring and "
                "access to academic resources."
            )
        else:
            st.success("✅ The model estimates lower academic risk.")
            st.write(
                "Continue regular progress monitoring and "
                "encourage healthy study habits."
            )

        if risk_probability is not None:
            st.metric(
                "Estimated At-Risk Probability",
                f"{risk_probability:.1%}"
            )
            st.progress(risk_probability)

        # -------------------------------
        # 6. EXPLAINABLE AI
        # -------------------------------
        st.divider()
        st.subheader("🔎 Explainable AI")
        st.write(
            "The following chart explains how input features "
            "contribute to the Logistic Regression model's score. "
            "These are model associations, not causes."
        )

        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        if hasattr(classifier, "coef_"):

            transformed = preprocessor.transform(student_df)
            feature_names = preprocessor.get_feature_names_out()

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            contributions = transformed[0] * classifier.coef_[0]

            explanation = pd.DataFrame({
                "Feature": feature_names,
                "Contribution": contributions
            })

            explanation = explanation[
                explanation["Contribution"] != 0
            ].copy()

            explanation["Direction"] = explanation[
                "Contribution"
            ].apply(
                lambda value: (
                    "Higher model score"
                    if value > 0
                    else "Lower model score"
                )
            )

            explanation["Magnitude"] = explanation[
                "Contribution"
            ].abs()

            explanation = explanation.sort_values(
                "Magnitude", ascending=False
            ).head(10)

            if not explanation.empty:

                st.caption(
                    "Positive contributions push the model score "
                    "toward the At Risk class; negative contributions "
                    "push it away from that class."
                )

                st.bar_chart(
                    explanation.set_index("Feature")["Contribution"]
                )

                st.dataframe(
                    explanation[
                        ["Feature", "Direction", "Contribution"]
                    ].round(3),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "No non-zero feature contributions were found "
                    "for these inputs."
                )

        else:
            st.info(
                "This model does not expose linear coefficients. "
                "The current explanation chart is designed for "
                "Logistic Regression."
            )

        st.warning(
            "This is an educational prototype trained on a small "
            "historical dataset. It is not a diagnosis or a definitive "
            "assessment of any student's ability or future."
        )

# -------------------------------
# 7. ABOUT PROJECT
# -------------------------------
elif page == "About Project":

    st.title("ℹ️ About EduPredict AI")

    st.write(
        "EduPredict AI is a machine learning prototype that "
        "estimates academic risk from student background, "
        "study habits, support and attendance information."
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Python** — core programming language
        - **Pandas** — data handling
        - **Scikit-learn** — preprocessing and machine learning
        - **Streamlit** — interactive web application
        - **Plotly / Streamlit charts** — visual analytics
        - **Joblib** — saved model pipeline
        """
    )

    st.subheader("Dataset and Target")

    st.write(
        "The prototype uses the UCI Student Performance dataset. "
        "Students with final grade G3 below 10 are labelled At Risk; "
        "others are labelled Not At Risk. G1, G2 and G3 are excluded "
        "from prediction inputs to reduce target leakage."
    )

    st.subheader("Limitations")

    st.write(
        "The model was trained on a small historical dataset and "
        "may not generalize to other schools or student populations. "
        "Its outputs should not be used alone for high-stakes "
        "educational decisions."
    )

    st.caption("EduPredict AI | Academic Risk Prediction Prototype")
