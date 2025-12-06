# ✈️ Apex Air: Network Operations Control (NOC) - Flight Delay Prediction

![Apex Air Logo](apex_air.png)

## 📌 Project Overview
**Apex Air** is a data science initiative designed to tackle one of the most critical challenges in the aviation industry: **Flight Delays**. 

Our goal is to build a machine learning system capable of predicting whether a scheduled flight will be delayed by more than **15 minutes**, at least **three hours before departure**. This "early warning system" empowers our Network Operations Control (NOC) team to take proactive measures—such as adjusting crew schedules, rerouting aircraft, or notifying passengers—thereby reducing operational costs and improving customer satisfaction.

## 💼 Business Problem
Flight delays are not just an inconvenience; they are a massive financial drain.
-   **Direct Costs**: Crew overtime, fuel burn, airport fines.
-   **Indirect Costs**: Missed connections, passenger compensation (vouchers), brand damage.
-   **The Opportunity**: By predicting delays accurately, we can intervene early. Our **Profit Curve Analysis** estimates that even a model with 63% accuracy can save **$400,000 annually** per 100,000 flights by optimizing crew and resource allocation.

## 📊 Data Source
We utilized a massive dataset of **7 Million domestic US flights from 2018**.
-   **Target Variable**: `LATE_15` (Binary: 1 if Arrival Delay > 15 mins, else 0).
-   **Key Features**:
    -   **Temporal**: `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR`.
    -   **Route**: `ORIGIN`, `DEST`, `OP_CARRIER` (Airline).
    -   **Operational**: `CRS_ELAPSED_TIME` (Scheduled Duration).

*Note: Initial experiments with granular weather data and "Tail Number" tracking were conducted but excluded from the final production model due to data sparsity and real-time availability constraints.*

## 🛠️ Methodology

### 1. Data Preprocessing
To handle the scale and complexity of the data, we built a robust pipeline:
-   **Target Encoding**: Applied to high-cardinality features (`ORIGIN`, `DEST`, `OP_CARRIER`) to capture the historical "risk" of each airport and airline without creating thousands of dummy variables.
-   **Standard Scaling**: Applied to numerical features to ensure model stability.
-   **Class Balancing**: The dataset is imbalanced (~19% delays). We used `class_weight='balanced'` and stratified subsampling to ensure the model learns to detect delays effectively.

### 2. Model Selection
We evaluated multiple algorithms, including:
-   Logistic Regression (Baseline)
-   Random Forest & Extra Trees
-   **XGBoost (Champion)** 🏆

**Why XGBoost?**
It provided the best balance between **Recall** (catching delays) and **Precision** (avoiding false alarms), achieving an **ROC-AUC of 0.69**. It is also highly efficient for tabular data and supports missing value handling out-of-the-box.

## 📈 Key Results
-   **ROC-AUC**: 0.69
-   **Balanced Accuracy**: 63%
-   **Profitability**: The model maximizes profit at a probability threshold of **0.35**, significantly outperforming a "no-model" baseline.

## 📂 Project Structure
```bash
├── dashboard.py               # Main Streamlit Dashboard application
├── fix_delay_model.py         # Core script for data processing & model training
├── analyze_airport_delays.py  # Script to generate specific delay cause analysis
├── requirements.txt           # Python dependencies
├── 2018.csv                   # Raw Dataset (Not included in repo due to size)
├── apex_air.png               # Project Logo
├── nyu_logo.png               # University Logo
├── NOC.png                    # Operations Center Image
├── advanced_plots/            # Generated EDA and Result plots
│   ├── profit_curve.png
│   ├── top10_airports_delay_causes.png
│   └── ...
└── stored_models/             # Serialized Models & Preprocessors
    ├── XGBoost.pkl
    └── preprocessors.pkl
```

## 🚀 How to Run Locally

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/rushxbh910/Apex_Air-Network-Operations-Control.git
    cd Apex_Air-Network-Operations-Control
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:
    ```bash
    streamlit run dashboard.py
    ```

## 🔮 Future Scope
-   **Real-Time API**: Wrap the model in a FastAPI service for integration with airline scheduling software.
-   **Graph Neural Networks**: Model the propagation of delays across the network using tail-number tracking.
-   **LLM Integration**: Parse unstructured NOTAMs (Notices to Air Missions) for real-time hazard detection.

---
**Author**: Apex Air Data Science Team
**University**: NYU Stern School of Business
