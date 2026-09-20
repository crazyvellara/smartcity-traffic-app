import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart City Traffic - Chicago", page_icon="🚦", layout="wide")

DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
        5: "Thursday", 6: "Friday", 7: "Saturday"}


@st.cache_resource
def load(path):
    return joblib.load(path)


@st.cache_data
def load_segments():
    return pd.read_csv("data/segments.csv")


def align(X, model):
    cols = getattr(model, "feature_names_in_", None)
    return X[list(cols)] if cols is not None else X


tt_model = load("models/travel_time_final_model.pkl")
cg_model = load("models/congestion_prediction_final_model.pkl")
cg_enc = load("models/congestion_street_encoder.pkl")
ar_model = load("models/accident_risk_final_model.pkl")
ar_enc = load("models/accident_risk_encoders.pkl")
segs = load_segments()

st.title("🚦 Smart City Traffic Predictions")
st.caption("Chicago · three machine-learning models trained on 2024 open data")

tab1, tab2, tab3 = st.tabs(["⏱️ Travel Time", "🚗 Congestion", "⚠️ Accident Risk"])

# ---------------- TAB 1: TRAVEL TIME ----------------
with tab1:
    st.subheader("Estimate ride travel time")
    c1, c2 = st.columns(2)
    with c1:
        miles = st.slider("Trip distance (miles)", 0.5, 25.0, 5.0, 0.5, key="tt_miles")
        hour = st.slider("Pickup hour (0-23)", 0, 23, 17, key="tt_hour")
    with c2:
        pu = st.number_input("Pickup community area (1-77)", 1, 77, 8, key="tt_pu")
        do = st.number_input("Dropoff community area (1-77)", 1, 77, 32, key="tt_do")

    if st.button("Estimate travel time", key="tt_btn"):
        X = pd.DataFrame([{
            "trip_miles": miles,
            "trip_hour": hour,
            "pickup_community_area": pu,
            "dropoff_community_area": do,
        }])
        X = align(X, tt_model)
        minutes = tt_model.predict(X)[0] / 60
        st.metric("Estimated trip time", f"{minutes:.1f} minutes")

# ---------------- TAB 2: CONGESTION ----------------
with tab2:
    st.subheader("Predict road congestion")
    known = set(cg_enc.classes_)
    streets = sorted(s for s in segs["street"].dropna().unique() if s in known)
    street = st.selectbox("Street", streets, key="cg_street")

    sub = segs[segs["street"] == street].reset_index(drop=True)
    labels = {int(r.segment_id): f"{r.from_street} → {r.to_street} ({r.direction})"
              for r in sub.itertuples()}
    seg_id = st.selectbox("Segment", list(labels.keys()),
                          format_func=lambda i: labels[i], key="cg_seg")

    c1, c2 = st.columns(2)
    with c1:
        hour = st.slider("Hour of day (0-23)", 0, 23, 17, key="cg_hour")
    with c2:
        dow = st.selectbox("Day of week", list(DAYS.keys()),
                           format_func=lambda d: DAYS[d], key="cg_dow")

    if st.button("Predict congestion", key="cg_btn"):
        row = sub[sub["segment_id"] == seg_id].iloc[0]
        X = pd.DataFrame([{
            "segment_id": seg_id,
            "hour": hour,
            "day_of_week": dow,
            "street": int(cg_enc.transform([street])[0]),
            "length": float(row["length"]),
        }])
        X = align(X, cg_model)
        p = cg_model.predict_proba(X)[0][1]
        st.metric("Chance traffic is below 15 mph", f"{p:.0%}")
        st.progress(float(p))
        st.caption("Trained on one week of June 2024 Chicago Traffic Tracker data.")

# ---------------- TAB 3: ACCIDENT RISK ----------------
with tab3:
    st.subheader("Predict crash injury risk")

    def pick(label, name, default=None):
        options = list(ar_enc[name].classes_)
        idx = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=idx, key="ar_" + name)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Conditions**")
        weather = pick("Weather", "weather_condition", "CLEAR")
        lighting = pick("Lighting", "lighting_condition", "DAYLIGHT")
        surface = pick("Road surface", "roadway_surface_cond", "DRY")
        control = pick("Traffic control", "traffic_control_device", "NO CONTROLS")
        trafficway = pick("Trafficway type", "trafficway_type")
        speed = st.slider("Posted speed limit (mph)", 5, 70, 30, 5, key="ar_speed")
    with c2:
        st.markdown("**Crash details**")
        first_type = pick("First crash type", "first_crash_type")
        cause = pick("Primary cause", "prim_contributory_cause")
        driver_action = pick("Driver action", "driver_action")
        condition = pick("Driver physical condition", "physical_condition", "NORMAL")
        safety = pick("Safety equipment", "safety_equipment", "SAFETY BELT USED")
        units = st.number_input("Number of vehicles/units", 1, 10, 2, key="ar_units")
    with c3:
        st.markdown("**Time and weather data**")
        crash_hour = st.slider("Crash hour (0-23)", 0, 23, 17, key="ar_hour")
        crash_dow = st.selectbox("Crash day of week", list(DAYS.keys()),
                                 format_func=lambda d: DAYS[d], key="ar_dow")
        crash_month = st.slider("Month", 1, 12, 6, key="ar_month")
        tmax = st.number_input("Max temperature (°C)", -30.0, 50.0, 22.0, key="ar_tmax")
        tmin = st.number_input("Min temperature (°C)", -30.0, 40.0, 12.0, key="ar_tmin")
        prcp = st.number_input("Rainfall (mm)", 0.0, 200.0, 0.0, key="ar_prcp")
        snow = st.number_input("Snowfall (mm)", 0.0, 500.0, 0.0, key="ar_snow")

    if st.button("Predict injury risk", key="ar_btn"):
        row = {
            "weather_condition": weather,
            "lighting_condition": lighting,
            "roadway_surface_cond": surface,
            "traffic_control_device": control,
            "crash_hour": crash_hour,
            "crash_day_of_week": crash_dow,
            "crash_month": crash_month,
            "posted_speed_limit": speed,
            "TMAX": tmax,
            "TMIN": tmin,
            "PRCP": prcp,
            "SNOW": snow,
            "safety_equipment": safety,
            "driver_action": driver_action,
            "physical_condition": condition,
            "first_crash_type": first_type,
            "prim_contributory_cause": cause,
            "num_units": units,
            "trafficway_type": trafficway,
        }
        X = pd.DataFrame([row])
        for col, enc in ar_enc.items():
            X[col] = enc.transform(X[col])
        X = align(X, ar_model)
        p = ar_model.predict_proba(X)[0][1]
        st.metric("Injury risk score", f"{p:.0%}")
        st.progress(float(p))
        st.caption("The model was trained with class balancing, so treat this as a relative "
                   "risk score, not a literal probability.")