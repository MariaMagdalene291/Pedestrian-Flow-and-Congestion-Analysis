# Pedestrian Flow and Congestion Analysis using PySpark

## Project Overview

This project analyzes pedestrian movement data collected from traffic sensors across Melbourne using Apache Spark (PySpark). The main objective is to identify pedestrian traffic patterns, detect congestion-prone locations, analyze peak hours, and build a simple machine learning model to predict pedestrian counts.

Since the dataset contains a large number of records collected over several years, PySpark was used to efficiently process and analyze the data.

---

## Objectives

* Analyze pedestrian traffic collected from multiple sensors.
* Identify peak pedestrian hours.
* Compare weekday and weekend pedestrian flow.
* Find congestion-prone sensor locations.
* Perform feature engineering for traffic analysis.
* Build a Linear Regression model to predict hourly pedestrian counts.

---

## Technologies Used

* Python
* Apache Spark (PySpark)
* Spark SQL
* Spark MLlib
* Pandas
* Matplotlib
* Google Colab

---

## Dataset

**Dataset:** Melbourne Pedestrian Counting System

The dataset contains hourly pedestrian counts recorded by sensors installed across Melbourne city.

Main attributes include:

* Sensor ID
* Sensor Name
* Date and Time
* Hour
* Day
* Hourly Counts

---

## Project Workflow

### 1. Data Loading

The CSV dataset is loaded into a Spark DataFrame using PySpark.

### 2. Data Preprocessing

The preprocessing stage includes:

* Handling missing values
* Removing duplicate records
* Converting date and time into timestamp format
* Filtering invalid timestamps

### 3. Feature Engineering

Several new features were created to improve analysis:

* Peak Hour Indicator
* Weekend Indicator
* Day of Year
* Rolling Average of Hourly Counts
* Previous Hour Count
* Average Flow per Sensor

These features help capture temporal traffic patterns and congestion trends.

### 4. Machine Learning

A Linear Regression model from Spark MLlib was trained using the engineered features.

**Features used:**

* Hour
* Peak Hour
* Weekend
* Rolling Hourly Count
* Previous Hour Count

**Target variable:**

* Hourly Counts

The model predicts expected pedestrian counts for each hour.

### 5. Data Analysis using Spark SQL

Spark SQL was used to answer several analytical questions, including:

* Which hours have the highest pedestrian flow?
* How does weekend traffic compare with weekdays?
* Which sensor locations experience the highest pedestrian traffic?
* How does traffic differ during peak and non-peak hours?

---

## Visualizations

The project includes the following visualizations using Matplotlib:

* Peak Hour Pedestrian Flow
* Weekday vs Weekend Comparison
* Top 10 Congestion-Prone Locations

These visualizations help in understanding pedestrian movement patterns and identifying busy locations.

---

## Results

The analysis showed that:

* Pedestrian traffic is highest during morning and evening peak hours.
* Some sensor locations consistently experience higher pedestrian volumes than others.
* Weekend pedestrian patterns differ from weekday traffic.
* Rolling averages and previous hour counts contribute to better traffic prediction.

---

## Project Structure

```text
Pedestrian-Flow-and-Congestion-Analysis/
│
├── pedestrian_flow_analysis.ipynb
├── mel_pedestrian2.csv
├── README.md
└── images/
```

---

## Future Improvements

Possible enhancements include:

* Using advanced machine learning models such as Random Forest Regression or Gradient Boosting.
* Building a real-time traffic monitoring dashboard using Streamlit.
* Integrating live sensor data for real-time congestion prediction.
* Creating interactive visualizations using Plotly or Power BI.

---

## Learning Outcomes

This project provided practical experience in:

* Big Data processing using Apache Spark
* Data cleaning and preprocessing
* Feature engineering
* Spark SQL
* Machine Learning with Spark MLlib
* Data visualization
* Traffic data analysis

---

## Author

**Maria Magdalene**

M.Sc. Data Science

This project was developed as part of my Big Data Analytics coursework to demonstrate how PySpark can be used for large-scale pedestrian traffic analysis and predictive modeling.
