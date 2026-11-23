# 🚦 Smart City Traffic Prediction System

An interactive machine learning-based web application for predicting traffic conditions in Chicago, including estimated travel time, road congestion, and accident injury risk.

The application integrates three trained machine learning models into a single Streamlit dashboard, allowing users to explore traffic predictions through an interactive interface.

## 📌 Project Overview

Urban traffic management is a major challenge for modern cities. Traffic congestion, unpredictable travel times, and road accidents affect transportation efficiency and public safety.

**Smart City Traffic Prediction System** explores how machine learning can support smarter urban transportation by analyzing traffic-related data and generating predictive insights.

The application currently focuses on three prediction modules:

- Travel Time Prediction
- Traffic Congestion Prediction
- Accident Injury Risk Prediction

## ✨ Key Features

### 1. ⏱️ Travel Time Prediction
Estimates the expected duration of a trip based on:
- Trip distance (miles)
- Pickup hour
- Pickup community area
- Drop-off community area

The trained regression model generates an estimated travel time in minutes.

### 2. 🚗 Traffic Congestion Prediction
Estimates the likelihood of traffic moving below 15 mph on a selected road segment.

**Input features include:**
- Street name
- Road segment
- Hour of the day
- Day of the week
- Road segment length

The application displays the model's predicted score through an interactive metric and progress indicator.

### 3. ⚠️ Accident Injury Risk Prediction
Analyzes road, environmental, and crash-related conditions to estimate accident injury risk.

**Input features include:**
- Weather conditions
- Lighting conditions
- Road surface conditions
- Traffic control devices
- Trafficway type
- Posted speed limit
- Crash type and contributing causes
- Driver action and physical condition
- Safety equipment
- Number of vehicles involved
- Temperature, rainfall, and snowfall
- Time and date of the incident

The model produces a relative injury risk score.

**Note:** The accident risk model uses class balancing. Its output should be interpreted as a relative risk score, not a calibrated real-world probability.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Visualization | Streamlit Metrics and Progress Indicators |
| Version Control | Git and GitHub |

## 🧠 Machine Learning Models

The application integrates three independently trained machine learning models:

| Model | Task | Output |
|---|---|---|
| Travel Time Model | Regression | Estimated trip duration |
| Congestion Model | Classification | Likelihood of low-speed traffic |
| Accident Risk Model | Classification | Relative injury risk score |

The trained models and their associated encoders are stored in the `models/` directory and loaded by the application at runtime.

## 🏗️ Project Structure

```text
smartcity-traffic-app/
│
├── app.py
│
├── requirements.txt
│
├── data/
│   └── segments.csv
│
├── models/
│   ├── travel_time_final_model.pkl
│   ├── congestion_prediction_final_model.pkl
│   ├── congestion_street_encoder.pkl
│   ├── accident_risk_final_model.pkl
│   └── accident_risk_encoders.pkl
│
└── README.md
```

## ⚙️ Installation and Setup

Follow these steps to run the application locally.

### 1. Clone the repository

```bash
git clone https://github.com/crazyvellara/smartcity-traffic-app.git
```

### 2. Navigate to the project directory

```bash
cd smartcity-traffic-app
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your default browser, usually at:

```text
http://localhost:8501
```

## 📊 Dataset

The project uses Chicago-related traffic and crash data.

The application includes traffic segment information in `data/segments.csv` and uses trained model artifacts stored in the `models/` directory.

The application describes its travel time and congestion models as being trained using 2024 data.

## 🖥️ Application Modules

The dashboard provides three interactive tabs:

1. Travel Time: Enter trip details to estimate journey duration.
2. Congestion: Select a street and road segment to evaluate traffic conditions.
3. Accident Risk: Configure environmental, road, and crash-related conditions to estimate injury risk.

## 🚀 Future Improvements

- Integrate real-time traffic data.
- Expand prediction coverage to additional urban regions.
- Improve prediction accuracy through model experimentation and evaluation.
- Integrate interactive traffic maps and geospatial visualizations.
- Explore optimized route recommendations using predicted traffic conditions.
- Develop a more comprehensive smart city traffic monitoring platform.

## 👩‍💻 Contributors

Developed as part of a Smart City Traffic Big Data Analytics and Predictive Traffic Intelligence project.

GitHub: [@crazyvellara](https://github.com/crazyvellara)

## 📄 License

This project is intended for educational and research purposes.
growthx code :2HGMWF

---

**Smart City Traffic Prediction System**  
Exploring the role of machine learning in building smarter urban transportation systems.
