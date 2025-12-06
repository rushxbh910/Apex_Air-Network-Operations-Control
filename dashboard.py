import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import category_encoders as ce
from datetime import datetime

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="Apex Air: Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_DIR = "."
PLOTS_DIR = os.path.join(DATA_DIR, "advanced_plots")
MODELS_DIR = os.path.join(DATA_DIR, "stored_models")

# Load Models & Preprocessors
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "XGBoost.pkl"))
        preprocessors = joblib.load(os.path.join(MODELS_DIR, "preprocessors.pkl"))
        return model, preprocessors
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, preprocessors = load_artifacts()

# ============================
# Sidebar Navigation
# ============================
st.sidebar.title("Apex Air ✈️")
st.sidebar.info("Network Operations Control (NOC)")

sections = [
    "1. Introduction",
    "2. Who Are We",
    "3. About the Data",
    "4. Exploratory Data Analysis (EDA)",
    "5. Data Preprocessing",
    "6. Model Pipeline & Selection",
    "7. Model Evaluation",
    "8. Profit Curve Analysis",
    "9. Real-Time Implementation",
    "10. Failed Experiments",
    "11. Future Scope"
]

selection = st.sidebar.radio("Navigate to Slide:", sections)

# ============================
# Content
# ============================

# 1. Introduction
if selection == "1. Introduction":
    st.title("✈️ Predicting Flight Delays for Major US Airlines")
    st.subheader("Business Problem")
    st.markdown("""
    **The Big Problem:**
    At Apex Air, we’re struggling with one of the toughest challenges in aviation: flight delays. Every delay doesn’t just affect a single flight; it sets off a domino effect across our entire network. Planes arrive late, crews fall out of rotation, connecting passengers miss their flights, and the entire schedule begins to unravel. The result is skyrocketing operational costs, exhausted staff, frustrated travelers, and a growing dent in our reputation for reliability.

    **The Goal (Data Science Task):**
    Our goal is to build a machine learning model capable of predicting whether a scheduled flight will be delayed by more than **15 minutes**, at least **three hours before its departure time**. With accurate, early predictions, our operations team can take proactive measures instead of reacting after the fact.
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(os.path.join(DATA_DIR, "apex_air.png"), use_container_width=True)

# 2. Who Are We
elif selection == "2. Who Are We":
    st.title("👨‍💻 Who Are We")
    st.markdown("""
    **Role:** Operations Analyst, Network Operations Control (NOC)
    
    **Our Role:**
    We work as an Operations Analyst in the Network Operations Control (NOC) center at Apex Air. Our role involves transforming real-time operational data into meaningful insights. Whether it’s weather fluctuations, aircraft rotations, or congestion at major hubs, we focus on helping our team make smarter, faster, data-driven decisions to keep the airline running smoothly.

    **Our Audience:**
    The primary users of this model are our **Operations Managers and Flight Dispatchers** in the NOC and other flight operations teams. They are responsible for the day-to-day management of hundreds of flights across our network. If a flight is predicted to have a high likelihood of delay, they can respond quickly by rerouting passengers, adjusting crew schedules, or assigning a backup aircraft. This foresight will improve punctuality, reduce costs, and greatly enhance the passenger experience.
    """)
    st.image(os.path.join(DATA_DIR, "NOC.png"), use_container_width=True, caption="NOC Operations Center")

# 3. About the Data
elif selection == "3. About the Data":
    st.title("📊 About the Data")
    st.markdown("""
    **Source:**
    -   **Flight Data (2018)**: Over 7 Million domestic US flights.
    -   **Weather Data**: Granular hourly weather events (Rain, Snow, Fog) - *Note: Used for initial analysis but excluded from final model due to sparsity.*
    
    **Key Features Used:**
    We focused on robust, high-availability schedule features:
    -   **Schedule**: `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR` (Departure Hour).
    -   **Route**: `ORIGIN`, `DEST` (Destination), `OP_CARRIER` (Airline).
    -   **Flight Info**: `CRS_ELAPSED_TIME` (Scheduled Duration).
    
    **Target Variable:**
    -   `LATE_15`: Binary Classification (1 if Arrival Delay > 15 mins, else 0).
    
    *Correction: Initial experiments included `HOURLY_TRAFFIC` and `DISTANCE`, but these were removed during feature selection due to redundancy or data pipeline constraints.*
    """)

# 4. EDA
elif selection == "4. Exploratory Data Analysis (EDA)":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Interactive Gallery of Delay Drivers.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("1. Target Distribution (Class Imbalance)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "1_target_distribution.png"), use_container_width=True)
            st.caption("Only ~19% of flights are late, creating a significant class imbalance.")
            
        with st.expander("2. Delay Causes (Top 10 Airports)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "top10_airports_delay_causes.png"), use_container_width=True)
            st.caption("Late Aircraft and Carrier Delays are the biggest drivers at major hubs.")

        with st.expander("3. Delay by Month (Seasonality)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "2_delay_by_month.png"), use_container_width=True)
            st.caption("Summer (Jun-Aug) and Winter Holidays (Dec) see the highest delay rates.")
            
        with st.expander("4. Delay by Day of Week", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "3_delay_by_day.png"), use_container_width=True)
            st.caption("Delays compound throughout the week, peaking on Friday.")
            
        with st.expander("5. Delay by Hour (Daily Pattern)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "4_delay_by_hour.png"), use_container_width=True)
            st.caption("Early morning flights are safe. Delays skyrocket after 3 PM.")

    with col2:
        with st.expander("6. Delay by Carrier", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "5_delay_by_carrier.png"), use_container_width=True)
            st.caption("Significant variance in operational reliability between airlines.")

        with st.expander("7. Delay by Origin (Hub Congestion)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "6_delay_by_origin.png"), use_container_width=True)
            st.caption("Major hubs like ORD and EWR suffer from chronic congestion.")
            
        with st.expander("8. Delay by Destination", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "7_delay_by_dest.png"), use_container_width=True)
            st.caption("Arrival capacity at destination is a key bottleneck.")
            
        with st.expander("9. Heatmap: Day vs Hour (Danger Zones)", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "9_heatmap_day_hour.png"), use_container_width=True)
            st.caption("Avoid flying Friday evenings! Tuesday mornings are the safest bet.")
            
        with st.expander("10. Correlation Matrix", expanded=True):
            st.image(os.path.join(PLOTS_DIR, "11_correlation_matrix.png"), use_container_width=True)
            st.caption("Departure Hour is the strongest temporal predictor.")

# 5. Data Preprocessing
elif selection == "5. Data Preprocessing":
    st.title("⚙️ Data Preprocessing")
    st.markdown("""
    To prepare the raw data for our machine learning pipeline, we applied the following techniques (reference: `fix_delay_model.py`):
    
    1.  **Target Encoding (Categorical Features)**:
        -   **Features**: `ORIGIN`, `DEST`, `OP_CARRIER`.
        -   **Technique**: We replaced each category (e.g., "JFK") with the *average probability of delay* for that category.
        -   **Why?**: These features have high cardinality (hundreds of airports). One-Hot Encoding would create a sparse matrix with thousands of columns, slowing down training. Target Encoding captures the "risk signal" of each airport efficiently.
        
    2.  **Standard Scaling (Numerical Features)**:
        -   **Features**: `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR`, `CRS_ELAPSED_TIME`.
        -   **Technique**: Transformed data to have Mean = 0 and Variance = 1.
        -   **Why?**: Essential for models like Logistic Regression to converge and ensures features with larger ranges (like Elapsed Time) don't dominate the model.
        
    3.  **Class Balancing**:
        -   **Technique**: `class_weight='balanced'`.
        -   **Why?**: Since only 19% of flights are late, a standard model would be biased towards predicting "On Time". We forced the model to pay equal attention to the minority "Late" class.
    """)

# 6. Model Pipeline
elif selection == "6. Model Pipeline & Selection":
    st.title("🤖 Model Pipeline")
    st.markdown("""
    We built a robust multi-model pipeline to identify the best algorithm for this specific task.
    
    **Models Evaluated:**
    1.  **Logistic Regression**: A simple, interpretable baseline.
    2.  **Decision Tree**: Captures non-linear decision boundaries.
    3.  **Random Forest**: An ensemble of trees to reduce overfitting.
    4.  **Extra Trees**: Randomized trees for lower variance.
    5.  **AdaBoost**: Adaptive boosting to focus on hard-to-classify examples.
    6.  **HistGradientBoosting**: A fast, histogram-based gradient boosting method (similar to LightGBM).
    7.  **XGBoost**: The industry standard for tabular classification.
    
    **Training Strategy:**
    -   **Split**: 80% Training, 20% Testing.
    -   **Subsampling (Why 500k?)**:
        -   **Efficiency vs. Accuracy**: We trained on a stratified sample of **500,000 flights** (approx. 7% of the data).
        -   **Justification**: Learning curves showed that model performance plateaued after ~300k samples. Using the full 7 million records would increase training time by 14x (from minutes to hours) without a statistically significant gain in AUC. This allowed us to iterate rapidly and tune hyperparameters effectively on local hardware.
    -   **Metric**: Optimized for **Balanced Accuracy** (Average of Recall and Specificity).
    """)

# 7. Model Evaluation
elif selection == "7. Model Evaluation":
    st.title("📈 Model Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(os.path.join(PLOTS_DIR, "model_comparison_all_metrics.png"), use_container_width=True, caption="Performance Comparison")
        
    with col2:
        st.markdown("""
        **The Winner: XGBoost** 🏆
        
        **Why XGBoost?**
        1.  **Best Balance**: It achieved the highest **Balanced Accuracy (63%)** and **ROC-AUC (0.69)**.
        2.  **Profitability**: Unlike AdaBoost (which had high accuracy but 0 Recall), XGBoost found the optimal trade-off between catching delays and avoiding false alarms.
        3.  **Robustness**: It handled the class imbalance better than any other model, maintaining a **Recall of ~60%** (catching 6 out of 10 delays).
        """)
        
    st.image(os.path.join(PLOTS_DIR, "calibration_curve.png"), caption="Calibration Curve: XGBoost probabilities are reliable.")

# 8. Profit Curve
elif selection == "8. Profit Curve Analysis":
    st.title("💰 Profit Curve Analysis")
    
    st.markdown("""
    **Real-Life Situation:**
    Imagine a Friday evening flight from JFK to LAX.
    -   **Scenario A (Missed Delay)**: The model predicts "On Time". We do nothing. The flight is delayed by 2 hours. Passengers miss connections, crew goes into overtime, and we have to issue vouchers. **Cost: -$100**.
    -   **Scenario B (False Alarm)**: The model predicts "Late". We call in a backup crew. The flight ends up leaving on time. We wasted money on the backup crew. **Cost: -$20**.
    -   **Scenario C (Success)**: The model predicts "Late". We prep the backup crew. The original crew times out, but the backup is ready. The flight leaves on time. We avoided the \$100 cost. **Net Benefit: +\$80**.
    
    **The Curve:**
    The Profit Curve below calculates the total expected profit for Apex Air at different "intervention thresholds".
    """)
    
    st.image(os.path.join(PLOTS_DIR, "profit_curve.png"), use_container_width=True)
    
    st.success("""
    **Strategic Insight:**
    The curve peaks at **0.35**. This means we should **NOT** use the standard 50% probability cutoff.
    
    **Conclusion:**
    In a real-world scenario with **100,000 flights a year**, this "mediocre" model (63% accuracy) is worth **$400,000 in pure savings**. That is a massive success, not a failure.
    """)

# 9. Real-Time Implementation
elif selection == "9. Real-Time Implementation":
    st.title("⚡ Real-Time Inference")
    st.markdown("Predict delay probability for a new flight using the **XGBoost** model.")
    
    if model is None:
        st.error("Model not found. Please run the training pipeline first.")
    else:
        # Input Form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                month = st.selectbox("Month", range(1, 13))
                day = st.selectbox("Day of Week", range(1, 8), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])
                hour = st.slider("Departure Hour", 0, 23, 17)
                
                # Carrier Dropdown
                carriers = ['UA', 'AA', 'DL', 'WN', 'B6', 'NK', 'AS', 'F9', 'HA', 'VX']
                carrier = st.selectbox("Carrier Code", carriers)
                
            with col2:
                # Airport Dropdown (Top 50 US Airports)
                airports = sorted(['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'LAS', 'SEA', 'CLT', 
                                 'EWR', 'MCO', 'PHX', 'MIA', 'IAH', 'BOS', 'MSP', 'DTW', 'FLL', 'PHL', 
                                 'LGA', 'BWI', 'SLC', 'SAN', 'IAD', 'DCA', 'MDW', 'TPA', 'PDX', 'HNL',
                                 'BNA', 'AUS', 'STL', 'RDU', 'SJC', 'OAK', 'SMF', 'MCI', 'MSY', 'SNA',
                                 'SAT', 'RSW', 'CLE', 'IND', 'PIT', 'CVG', 'CMH', 'OGG', 'PBI', 'BDL'])
                                 
                origin = st.selectbox("Origin Airport", airports, index=airports.index("JFK"))
                dest = st.selectbox("Destination Airport", airports, index=airports.index("LAX"))
                elapsed = st.number_input("Scheduled Duration (mins)", 360)
                
            submitted = st.form_submit_button("Predict Delay")
            
        if submitted:
            # Prepare Data
            input_data = pd.DataFrame({
                "MONTH": [month],
                "DAY_OF_WEEK": [day],
                "DEP_HOUR": [hour],
                "OP_CARRIER": [carrier],
                "ORIGIN": [origin],
                "DEST": [dest],
                "CRS_ELAPSED_TIME": [elapsed]
            })
            
            # Preprocess
            try:
                encoder = preprocessors["encoder"]
                scaler = preprocessors["scaler"]
                num_cols = preprocessors["num_cols"]
                
                # Encode
                input_enc = encoder.transform(input_data)
                
                # Scale
                input_scaled = input_enc.copy()
                input_scaled[num_cols] = scaler.transform(input_enc[num_cols])
                
                # Predict
                proba = model.predict_proba(input_enc)[:, 1][0]
                
                st.metric("Delay Probability", f"{proba:.1%}")
                
                # Profit Curve Threshold
                THRESHOLD = 0.35
                
                if proba > THRESHOLD:
                    st.error(f"⚠️ High Risk of Delay! (Probability > {THRESHOLD*100:.0f}%)")
                    st.info(f"**Recommendation:** Activate proactive crew management. Based on our Profit Curve, intervening here saves money.")
                else:
                    st.success("✅ Low Risk. Flight likely on time.")
                    
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                st.warning("Note: Ensure Carrier/Airport codes exist in training data.")

# 10. Failed Experiments
elif selection == "10. Failed Experiments":
    st.title("🧪 Failed Experiments")
    st.markdown("""
    Science is about trial and error. We explored several avenues that ultimately did not make it into the final model.
    
    **Attempt 1: Two-Model Approach (Divide & Conquer)**
    -   **Idea**: We initially tried to build two separate models:
        1.  **Model A**: Predict non-weather delays (Carrier, NAS, Security) using schedule data.
        2.  **Model B**: Predict `WEATHER_DELAY` specifically using granular weather data.
        3.  **Merge**: Combine predictions to get a total delay probability.
    -   **Result**: **Failure**. The `WEATHER_DELAY` target was extremely sparse (>90% Null/Zero). Model B failed to learn anything meaningful, and Model A suffered from data leakage when we tried to use delay-cause columns that are only available *post-flight*.
    
    **Attempt 2: Predicting Weather Delays Only**
    -   **Idea**: Focus specifically on the `WEATHER_DELAY` column as the target.
    -   **Result**: **Failure**. More than 90% of the data had null values for this column, and of the remaining 10%, most were 0. The data was incredibly skewed and sparse, making it impossible for the model to learn meaningful patterns.
    
    **Attempt 3: External Weather Integration**
    -   **Idea**: Merge external weather data (Precipitation, Severity) from `WeatherEvents.csv` based on Origin and Destination timestamps.
    -   **Result**: **Failure**. While conceptually sound, the weather events were sparse (most hours have "No Event"). The resulting features were mostly 0 or Null. The added complexity of the data pipeline did not yield a significant improvement in AUC compared to the robust schedule-based features.
    
    **Technical Conclusion:**
    We prioritized **Data Availability** and **Model Robustness** over theoretical feature richness. The schedule-based features (`DEP_HOUR`, `OP_CARRIER`, etc.) provide a dense, high-signal representation of the operational state that is guaranteed to be available at the inference horizon (T-3 hours). In contrast, the weather and delay-propagation features suffered from extreme sparsity (>90% zeros) and data leakage risks, introducing noise that degraded generalization performance on the test set.
    """)

# 11. Future Scope
elif selection == "11. Future Scope":
    st.title("🚀 Future Scope")
    st.markdown("""
    To take this project to the next level, we propose the following technical enhancements:
    
    1.  **Tail Number Propagation (Graph Network)**:
        -   **Concept**: The single biggest predictor of a delay is the *incoming aircraft* being late.
        -   **Tech**: Build a Directed Acyclic Graph (DAG) of flight connections using `TAIL_NUM`.
        -   **Implementation**: Use Graph Neural Networks (GNNs) or simply feature engineer `PREV_ARR_DELAY` by linking the current flight to the previous leg of the same aircraft.
        
    2.  **LLM-Powered NOTAM Analysis**:
        -   **Concept**: Pilots receive "Notices to Air Missions" (NOTAMs) containing critical unstructured text about runway closures, bird hazards, and VIP movement.
        -   **Tech**: Use Large Language Models (LLMs) to parse these text blobs in real-time and extract binary risk flags (`is_runway_closed`, `is_vip_movement`).
        
    3.  **Real-Time API Deployment**:
        -   **Concept**: Integrate the model directly into the NOC's dashboard software.
        -   **Tech**: Wrap the model in a **FastAPI** microservice, containerize it with **Docker**, and deploy it to a Kubernetes cluster for high-availability inference.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Apex Air Data Science Team")

# NYU Logo at Bottom Right
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col2:
    st.image(os.path.join(DATA_DIR, "nyu_logo.png"), use_container_width=True)
