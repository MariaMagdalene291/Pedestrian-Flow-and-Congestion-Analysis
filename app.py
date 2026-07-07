import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, lag, expr, dayofyear
from pyspark.sql.window import Window

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# PAGE CONFIG
st.set_page_config(page_title="Smart City Pedestrian Dashboard", layout="wide")

st.title("🌆 Smart City Pedestrian Flow & Congestion Monitoring System")

# CACHE SPARK SESSION
@st.cache_resource
def create_spark():
    return SparkSession.builder.appName("PedestrianApp").getOrCreate()

spark = create_spark()

# CACHE DATA LOADING + PROCESSING
@st.cache_data
def load_data():

    df = spark.read.csv("mel_pedestrian2.csv", header=True, inferSchema=True)

    # Data Cleaning
    df = df.na.fill({"Hourly_Counts": 0, "Sensor_Name": "Unknown"})

    df = df.withColumn(
        "Date_Time",
        expr("try_to_timestamp(Date_Time, 'MMMM d, yyyy hh:mm:ss a')")
    )

    df = df.filter(col("Date_Time").isNotNull())
    df = df.dropDuplicates()

    # Feature Engineering
    df = df.withColumn(
        "Is_Peak_Hour",
        when((col("Time").between(7,9)) | (col("Time").between(17,19)), 1).otherwise(0)
    )

    df = df.withColumn(
        "Is_Weekend",
        when(col("Day").isin("Saturday","Sunday"),1).otherwise(0)
    )

    df = df.withColumn("DayOfYear", dayofyear(col("Date_Time")))

    windowSpec = Window.partitionBy("Sensor_ID").orderBy("Date_Time").rowsBetween(-1,1)
    df = df.withColumn("Rolling_Hourly_Count", avg("Hourly_Counts").over(windowSpec))

    windowSpec2 = Window.partitionBy("Sensor_ID").orderBy("Date_Time")
    df = df.withColumn("Prev_Hour_Count", lag("Hourly_Counts").over(windowSpec2))

    df = df.na.fill({
        "Rolling_Hourly_Count": 0,
        "Prev_Hour_Count": 0
    })

    # Convert safely to Pandas (limit for performance)
    return df.limit(50000).toPandas()

pdf = load_data()

# KPI SECTION
total_sensors = pdf["Sensor_ID"].nunique()
avg_flow = pdf["Hourly_Counts"].mean()

col1, col2 = st.columns(2)
col1.metric("📍 Total Sensors", total_sensors)
col2.metric("🚶 Average Hourly Flow", round(avg_flow, 2))

st.divider()

# SIDEBAR CONTROLS
st.sidebar.header("🔎 Interactive Controls")

sensor_list = sorted(pdf["Sensor_Name"].unique())
selected_sensor = st.sidebar.selectbox("Select Sensor Location", sensor_list)

min_date = pdf["Date_Time"].min()
max_date = pdf["Date_Time"].max()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

threshold = st.sidebar.slider("Congestion Alert Threshold", 0, 2000, 500)

# Filtered Data
filtered_pdf = pdf[
    (pdf["Sensor_Name"] == selected_sensor) &
    (pdf["Date_Time"] >= pd.to_datetime(start_date)) &
    (pdf["Date_Time"] <= pd.to_datetime(end_date))
]

st.subheader(f"📊 Filtered Data for {selected_sensor}")
st.dataframe(filtered_pdf.head())

# CONGESTION ALERT
avg_selected = filtered_pdf["Hourly_Counts"].mean()

if avg_selected > threshold:
    st.error("🚨 High Congestion Detected!")
else:
    st.success("✅ Traffic Flow Normal")

# TREND LINE
st.subheader("📈 Pedestrian Flow Trend")

fig_trend = plt.figure(figsize=(10,4))
plt.plot(filtered_pdf["Date_Time"], filtered_pdf["Hourly_Counts"])
plt.xticks(rotation=45)
plt.ylabel("Hourly Counts")
st.pyplot(fig_trend)

# PEAK HOUR ANALYSIS
st.subheader("🔥 Peak Hour Analysis")

peak_hour_df = pdf.groupby("Time")["Hourly_Counts"].mean()

fig1 = plt.figure(figsize=(8,4))
plt.bar(peak_hour_df.index, peak_hour_df.values)
plt.xlabel("Hour")
plt.ylabel("Average Pedestrian Count")
st.pyplot(fig1)

# TOP 10 SENSORS
st.subheader("🚦 Top 10 Congestion-Prone Locations")

top_sensors = pdf.groupby("Sensor_Name")["Hourly_Counts"].mean() \
    .sort_values(ascending=False).head(10)

fig2 = plt.figure(figsize=(8,5))
plt.barh(top_sensors.index, top_sensors.values)
plt.gca().invert_yaxis()
st.pyplot(fig2)

# MACHINE LEARNING (CACHED)
@st.cache_resource
def train_model(data):

    ml_data = data[[
        "Time",
        "Is_Peak_Hour",
        "Is_Weekend",
        "Rolling_Hourly_Count",
        "Prev_Hour_Count",
        "Hourly_Counts"
    ]].dropna()

    X = ml_data[[
        "Time",
        "Is_Peak_Hour",
        "Is_Weekend",
        "Rolling_Hourly_Count",
        "Prev_Hour_Count"
    ]]

    y = ml_data["Hourly_Counts"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    return model, r2

model, r2 = train_model(pdf)

st.header("🤖 Pedestrian Flow Prediction Model")
st.metric("📈 Model R² Score", round(r2, 4))

# REAL-TIME PREDICTION
st.subheader("🔮 Predict Future Pedestrian Flow")

input_hour = st.number_input("Enter Hour (0-23)", 0, 23, 12)
input_weekend = st.selectbox("Weekend?", ["No", "Yes"])

is_peak = 1 if (7 <= input_hour <= 9 or 17 <= input_hour <= 19) else 0
is_weekend = 1 if input_weekend == "Yes" else 0

avg_selected_safe = avg_selected if not pd.isna(avg_selected) else 0

input_data = [[
    input_hour,
    is_peak,
    is_weekend,
    avg_selected_safe,
    avg_selected_safe
]]

prediction = model.predict(input_data)[0]

st.metric("Predicted Pedestrian Count", round(prediction, 2))

# DOWNLOAD BUTTON
csv = filtered_pdf.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Filtered Data",
    csv,
    "filtered_pedestrian_data.csv",
    "text/csv"
)

st.success("✅ Smart City Interactive Dashboard Fully Loaded!")