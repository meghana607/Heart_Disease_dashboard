
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Heart Disease Data Analytics Dashboard")
st.caption("UCI Heart Disease Dataset | Exploratory Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_cleaned.csv")

    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df["diagnosis_group"] = df["num"].apply(
        lambda x: "Zero diagnosis value"
        if x == 0 else "Nonzero diagnosis value"
    )
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset file not found. Upload "
        "heart_disease_cleaned.csv to this GitHub repository."
    )
    st.stop()

# Summary metrics
total = len(df)
avg_age = df["age"].mean()
zero_count = int((df["num"] == 0).sum())
nonzero_count = int((df["num"] > 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", total)
c2.metric("Average Age", f"{avg_age:.1f} years")
c3.metric("Diagnosis Value 0", zero_count)
c4.metric(
    "Nonzero Diagnosis",
    f"{nonzero_count / total * 100:.1f}%"
)

st.divider()

# Interactive filters
st.sidebar.header("Dashboard Filters")
age_min = int(df["age"].min())
age_max = int(df["age"].max())

age_range = st.sidebar.slider(
    "Select age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

filtered = df[
    df["age"].between(age_range[0], age_range[1])
]

st.subheader("Dataset Overview")
st.write(f"Showing **{len(filtered)}** records for the selected age range.")

left, right = st.columns(2)

with left:
    counts = (
        filtered["num"].value_counts()
        .sort_index()
        .rename_axis("Diagnosis Category")
        .reset_index(name="Records")
    )
    fig = px.bar(
        counts,
        x="Diagnosis Category",
        y="Records",
        text="Records",
        title="Records by Diagnosis Category"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        filtered,
        x="age",
        nbins=15,
        title="Age Distribution",
        labels={"age": "Age (years)"}
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig = px.box(
        filtered,
        x="diagnosis_group",
        y="thalach",
        color="diagnosis_group",
        title="Maximum Heart Rate by Diagnosis Group",
        labels={
            "diagnosis_group": "Diagnosis Group",
            "thalach": "Maximum Heart Rate"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        filtered,
        x="diagnosis_group",
        y="chol",
        color="diagnosis_group",
        title="Cholesterol by Diagnosis Group",
        labels={
            "diagnosis_group": "Diagnosis Group",
            "chol": "Cholesterol (mg/dl)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Age vs Maximum Heart Rate")
fig = px.scatter(
    filtered,
    x="age",
    y="thalach",
    color="diagnosis_group",
    title="Age and Maximum Heart Rate",
    labels={
        "age": "Age (years)",
        "thalach": "Maximum Heart Rate"
    }
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Heatmap")
numeric_df = filtered.select_dtypes(include="number")
corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Between Numeric Variables"
)
st.plotly_chart(fig, use_container_width=True)

# Findings based on the selected records
st.divider()
st.header("Key Findings")

if len(filtered) > 0:
    st.write(
        f"- The selected data contains {len(filtered)} records "
        f"with an average age of {filtered['age'].mean():.1f} years."
    )

    zero = int((filtered["num"] == 0).sum())
    nonzero = int((filtered["num"] > 0).sum())

    st.write(
        f"- {zero} records have diagnosis value 0, while "
        f"{nonzero} have a nonzero diagnosis value."
    )

    means = filtered.groupby("diagnosis_group")["thalach"].mean()
    if len(means) == 2:
        st.write(
            f"- Mean maximum heart rate: "
            f"{means['Zero diagnosis value']:.1f} for the "
            f"zero-value group and "
            f"{means['Nonzero diagnosis value']:.1f} for the "
            f"nonzero-value group."
        )

    chol_means = filtered.groupby("diagnosis_group")["chol"].mean()
    if len(chol_means) == 2:
        st.write(
            f"- Mean cholesterol: "
            f"{chol_means['Zero diagnosis value']:.1f} mg/dl "
            f"for the zero-value group and "
            f"{chol_means['Nonzero diagnosis value']:.1f} mg/dl "
            f"for the nonzero-value group."
        )

st.header("Conclusion")
st.write(
    "This dashboard explores patterns in age, cholesterol, maximum "
    "heart rate, and diagnosis categories in the UCI Heart Disease "
    "dataset. Observed differences describe this dataset and do not "
    "establish causation or diagnose individuals. This project "
    "performs exploratory data analysis and does not predict "
    "heart disease for new patients."
)

st.caption("Dataset: UCI Heart Disease | Python, Pandas, Plotly and Streamlit")

