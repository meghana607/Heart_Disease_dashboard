
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Analytics Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Data Analytics Dashboard")
st.caption(
    "UCI Heart Disease Dataset | Interactive Exploratory Data Analysis"
)

# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("heart_disease_cleaned.csv")

    # Convert numeric columns safely
  for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["ca", "thal"]:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])

    data["diagnosis_group"] = data["num"].apply(
        lambda x: "Diagnosis 0" if x == 0 else "Nonzero diagnosis"
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Upload heart_disease_cleaned.csv "
        "to the same GitHub repository as app.py."
    )
    st.stop()

if df.empty or "num" not in df.columns:
    st.error("The dataset is empty or does not contain the 'num' column.")
    st.stop()

# --------------------------------------------------
# SUMMARY CARDS - ALWAYS FULL DATASET
# --------------------------------------------------
total_records = len(df)
average_age = df["age"].mean()
zero_count = int((df["num"] == 0).sum())
nonzero_count = int((df["num"] > 0).sum())

st.subheader("Dataset at a Glance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Records", f"{total_records}")
m2.metric("Mean Age", f"{average_age:.1f} years")
m3.metric(
    "Diagnosis Value 0",
    f"{zero_count}",
    f"{zero_count / total_records * 100:.2f}% of records"
)
m4.metric(
    "Nonzero Diagnosis",
    f"{nonzero_count}",
    f"{nonzero_count / total_records * 100:.2f}% of records"
)

st.divider()

# --------------------------------------------------
# INTERACTIVE FILTERS
# --------------------------------------------------
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Change these filters to explore the data.")

age_min = int(df["age"].min())
age_max = int(df["age"].max())

age_range = st.sidebar.slider(
    "Select age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
    help="Move the handles to select the age group."
)

diagnosis_values = sorted(df["num"].dropna().unique().tolist())

selected_diagnoses = st.sidebar.multiselect(
    "Select diagnosis categories",
    options=diagnosis_values,
    default=diagnosis_values,
    help="Choose one or more diagnosis values."
)

filtered = df[
    df["age"].between(age_range[0], age_range[1])
    & df["num"].isin(selected_diagnoses)
].copy()

st.subheader("Explore the Dataset")

st.write(
    f"**{len(filtered)} records** match your selected filters "
    f"out of {total_records} total records."
)

if filtered.empty:
    st.warning("No records match these filters. Change the age range or diagnosis selection.")
    st.stop()

# --------------------------------------------------
# INTERACTIVE CHARTS
# --------------------------------------------------
left, right = st.columns(2)

with left:
    diagnosis_counts = (
        filtered["num"]
        .value_counts()
        .sort_index()
        .rename_axis("Diagnosis Value")
        .reset_index(name="Number of Records")
    )

    fig = px.bar(
        diagnosis_counts,
        x="Diagnosis Value",
        y="Number of Records",
        color="Diagnosis Value",
        text="Number of Records",
        title="Diagnosis Category Distribution",
        hover_data={"Number of Records": True}
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        filtered,
        x="age",
        nbins=15,
        color="diagnosis_group",
        title="Age Distribution",
        labels={"age": "Age (years)", "count": "Number of Records"},
        hover_data=["num"]
    )

    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig = px.box(
        filtered,
        x="num",
        y="thalach",
        color="num",
        points="all",
        title="Maximum Heart Rate by Diagnosis Value",
        labels={
            "num": "Diagnosis Value",
            "thalach": "Maximum Heart Rate"
        },
        hover_data=["age", "chol"]
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        filtered,
        x="num",
        y="chol",
        color="num",
        points="all",
        title="Cholesterol by Diagnosis Value",
        labels={
            "num": "Diagnosis Value",
            "chol": "Cholesterol (mg/dl)"
        },
        hover_data=["age", "thalach"]
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Age vs Maximum Heart Rate")

fig = px.scatter(
    filtered,
    x="age",
    y="thalach",
    color="diagnosis_group",
    symbol="num",
    hover_data=["chol", "num"],
    title="Explore Age and Maximum Heart Rate",
    labels={
        "age": "Age (years)",
        "thalach": "Maximum Heart Rate",
        "diagnosis_group": "Diagnosis Group",
        "num": "Diagnosis Value"
    }
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------
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

st.caption(
    "Correlation indicates association, not causation. Encoded diagnosis "
    "and category variables require careful interpretation."
)

# --------------------------------------------------
# FIVE IMPORTANT FINDINGS - FULL DATASET
# --------------------------------------------------
st.divider()
st.header("📌 Five Important Findings")
st.caption(
    "These findings use the complete dataset, independent of the filters above."
)

# Finding 1: dataset overview
st.markdown("### 1. Dataset Overview")
st.write(
    f"The dataset contains **{total_records} records** and "
    f"{df.shape[1] - 1} original feature columns, with a mean age "
    f"of **{average_age:.2f} years**."
)

# Finding 2: diagnosis distribution
st.markdown("### 2. Diagnosis Distribution")
st.write(
    f"**{zero_count} records ({zero_count / total_records * 100:.2f}%)** "
    f"have diagnosis value 0, while **{nonzero_count} records "
    f"({nonzero_count / total_records * 100:.2f}%)** have nonzero "
    f"diagnosis values."
)

# Finding 3: maximum heart rate, diagnosis 0 versus 1
st.markdown("### 3. Maximum Heart Rate Comparison")

mean_hr_0 = df.loc[df["num"] == 0, "thalach"].mean()
mean_hr_1 = df.loc[df["num"] == 1, "thalach"].mean()

if pd.notna(mean_hr_0) and pd.notna(mean_hr_1):
    st.write(
        f"Mean maximum heart rate is **{mean_hr_0:.2f}** for diagnosis "
        f"value 0 and **{mean_hr_1:.2f}** for diagnosis value 1."
    )
else:
    st.write("The mean maximum heart rate could not be calculated for both categories.")

# Finding 4: cholesterol, diagnosis 0 versus 2
st.markdown("### 4. Cholesterol Comparison")

mean_chol_0 = df.loc[df["num"] == 0, "chol"].mean()
mean_chol_2 = df.loc[df["num"] == 2, "chol"].mean()

if pd.notna(mean_chol_0) and pd.notna(mean_chol_2):
    st.write(
        f"Mean cholesterol is **{mean_chol_0:.2f} mg/dl** for diagnosis "
        f"value 0 and **{mean_chol_2:.2f} mg/dl** for diagnosis value 2."
    )
else:
    st.write("The mean cholesterol level could not be calculated for both categories.")

# Finding 5: comparison between zero and all nonzero diagnosis values
st.markdown("### 5. Zero vs Nonzero Diagnosis Groups")

zero_group = df[df["num"] == 0]
nonzero_group = df[df["num"] > 0]

if not zero_group.empty and not nonzero_group.empty:
    hr_zero = zero_group["thalach"].mean()
    hr_nonzero = nonzero_group["thalach"].mean()
    chol_zero = zero_group["chol"].mean()
    chol_nonzero = nonzero_group["chol"].mean()

    st.write(
        f"The mean maximum heart rate is **{hr_zero:.2f}** in the "
        f"zero-value group and **{hr_nonzero:.2f}** in the nonzero group."
    )
    st.write(
        f"Mean cholesterol is **{chol_zero:.2f} mg/dl** in the "
        f"zero-value group and **{chol_nonzero:.2f} mg/dl** in the nonzero group."
    )
else:
    st.write("Both groups are needed to make this comparison.")

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------
st.divider()
st.header("Conclusion")

st.write(
    "This dashboard provides interactive exploratory analysis of the UCI "
    "Heart Disease dataset. Filters and visualizations help explore patterns "
    "in age, cholesterol, maximum heart rate, and diagnosis categories. "
    "These results describe the dataset and do not establish causation or "
    "diagnose individuals. No machine-learning prediction model is used."
)

st.caption(
    "Tools: Python | Pandas | Plotly | Streamlit | Dataset: UCI Heart Disease"
)
