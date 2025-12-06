import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import joblib
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    AdaBoostClassifier, 
    HistGradientBoostingClassifier
)

# Try XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping.")

# ============================
# Configuration
# ============================
DATA_DIR = "/Users/rushabhbhatt/Documents/NYU_Fall2025/datascience_stern/data_science_project/final_proj/other_delay_features"
FLIGHTS_FILE = os.path.join(DATA_DIR, "2018.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "advanced_plots")
MODELS_DIR = os.path.join(DATA_DIR, "stored_models")
METRICS_FILE = os.path.join(DATA_DIR, "model_metrics.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

# Set Plot Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 100})

def load_data():
    print("Loading raw flight data...")
    cols = [
        "FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", 
        "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
        "ARR_DELAY", "CANCELLED", "DIVERTED"
    ]
    df = pd.read_csv(FLIGHTS_FILE, usecols=cols)
    
    # Filter
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    df = df.dropna(subset=["ARR_DELAY", "CRS_DEP_TIME", "CRS_ARR_TIME"])
    
    # Feature Engineering
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek + 1
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int)
    
    # Target
    df["LATE_15"] = (df["ARR_DELAY"] > 15).astype(int)
    
    print(f"Loaded {len(df)} flights.")
    return df

def run_detailed_eda(df, target="LATE_15"):
    print("Running Detailed EDA (10-12 Plots)...")
    
    # 1. Target Distribution
    plt.figure(figsize=(6,6))
    df[target].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], explode=(0,0.1))
    plt.title("Target Distribution: Late > 15 Minutes")
    plt.ylabel("")
    plt.savefig(os.path.join(PLOTS_DIR, "1_target_distribution.png"))
    plt.close()
    
    # 2. Delay by Month
    plt.figure()
    sns.barplot(x="MONTH", y=target, data=df, errorbar=None, palette="Blues_d")
    plt.title("Probability of Delay by Month")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(PLOTS_DIR, "2_delay_by_month.png"))
    plt.close()
    
    # 3. Delay by Day of Week
    plt.figure()
    sns.barplot(x="DAY_OF_WEEK", y=target, data=df, errorbar=None, palette="Greens_d")
    plt.title("Probability of Delay by Day of Week (1=Mon, 7=Sun)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(PLOTS_DIR, "3_delay_by_day.png"))
    plt.close()
    
    # 4. Delay by Hour
    plt.figure()
    sns.barplot(x="DEP_HOUR", y=target, data=df, errorbar=None, palette="Purples_d")
    plt.title("Probability of Delay by Departure Hour")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(PLOTS_DIR, "4_delay_by_hour.png"))
    plt.close()
    
    # 5. Delay by Carrier
    plt.figure(figsize=(12,6))
    order = df.groupby("OP_CARRIER")[target].mean().sort_values(ascending=False).index
    sns.barplot(x="OP_CARRIER", y=target, data=df, order=order, errorbar=None, palette="viridis")
    plt.title("Probability of Delay by Carrier")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(PLOTS_DIR, "5_delay_by_carrier.png"))
    plt.close()
    
    # 6. Delay by Origin (Top 20)
    plt.figure(figsize=(14,6))
    top_origins = df["ORIGIN"].value_counts().nlargest(20).index
    order_org = df[df["ORIGIN"].isin(top_origins)].groupby("ORIGIN")[target].mean().sort_values(ascending=False).index
    sns.barplot(x="ORIGIN", y=target, data=df[df["ORIGIN"].isin(top_origins)], order=order_org, errorbar=None, palette="magma")
    plt.title("Probability of Delay by Origin (Top 20 Busiest)")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, "6_delay_by_origin.png"))
    plt.close()
    
    # 7. Delay by Destination (Top 20)
    plt.figure(figsize=(14,6))
    top_dests = df["DEST"].value_counts().nlargest(20).index
    order_dest = df[df["DEST"].isin(top_dests)].groupby("DEST")[target].mean().sort_values(ascending=False).index
    sns.barplot(x="DEST", y=target, data=df[df["DEST"].isin(top_dests)], order=order_dest, errorbar=None, palette="magma")
    plt.title("Probability of Delay by Destination (Top 20 Busiest)")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, "7_delay_by_dest.png"))
    plt.close()
    
    # 8. Delay by Distance (Binning) - REMOVED due to redundancy
    # plt.figure()
    # df["Dist_Bin"] = pd.qcut(df["DISTANCE"], q=10)
    # sns.barplot(x="Dist_Bin", y=target, data=df, errorbar=None, palette="coolwarm")
    # plt.title("Probability of Delay by Flight Distance (Deciles)")
    # plt.xticks(rotation=45)
    # plt.savefig(os.path.join(PLOTS_DIR, "8_delay_by_distance.png"))
    # plt.close()
    
    # 9. Heatmap: Day vs Hour
    plt.figure(figsize=(12,6))
    pivot = df.pivot_table(index="DAY_OF_WEEK", columns="DEP_HOUR", values=target, aggfunc="mean")
    sns.heatmap(pivot, cmap="coolwarm", annot=False)
    plt.title("Heatmap: Delay Probability (Day vs Hour)")
    plt.savefig(os.path.join(PLOTS_DIR, "9_heatmap_day_hour.png"))
    plt.close()
    
    # 10. Heatmap: Origin vs Carrier (Top 10)
    plt.figure(figsize=(12,8))
    top_carr = df["OP_CARRIER"].value_counts().nlargest(10).index
    top_org = df["ORIGIN"].value_counts().nlargest(10).index
    pivot2 = df[df["OP_CARRIER"].isin(top_carr) & df["ORIGIN"].isin(top_org)].pivot_table(index="ORIGIN", columns="OP_CARRIER", values=target, aggfunc="mean")
    sns.heatmap(pivot2, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Heatmap: Delay Probability (Top Origins vs Top Carriers)")
    plt.savefig(os.path.join(PLOTS_DIR, "10_heatmap_origin_carrier.png"))
    plt.close()
    
    # 11. Correlation Matrix
    plt.figure(figsize=(10,8))
    numeric_cols = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR", "CRS_ELAPSED_TIME", target]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, "11_correlation_matrix.png"))
    plt.close()

    print(f"EDA Plots saved to {PLOTS_DIR}")

def calculate_profit_curve(y_true, y_proba, cost_benefit_matrix):
    # cost_benefit_matrix = [[TN, FP], [FN, TP]] (Benefit/Cost)
    # Usually:
    # TN: 0 (Status Quo)
    # FP: -Cost of Intervention (e.g. Crew Overtime prep that wasn't needed)
    # FN: -Cost of Delay (e.g. Rebooking, Loss of Goodwill)
    # TP: Benefit of Intervention - Cost of Intervention (e.g. Saved Rebooking - Prep Cost)
    
    thresholds = np.linspace(0, 1, 101)
    profits = []
    
    tn_val, fp_val = cost_benefit_matrix[0]
    fn_val, tp_val = cost_benefit_matrix[1]
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        profit = (tn * tn_val) + (fp * fp_val) + (fn * fn_val) + (tp * tp_val)
        profits.append(profit)
        
    return thresholds, profits

def train_and_evaluate_models(df, target="LATE_15"):
    # Features
    # Removed DISTANCE due to high correlation (0.98) with CRS_ELAPSED_TIME
    features = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR", "OP_CARRIER", "ORIGIN", "DEST", "CRS_ELAPSED_TIME"]
    X = df[features]
    y = df[target]
    
    # Split
    # Subsample for speed if needed, but let's try full dataset or large sample
    # 7M rows is a lot for some models. Let's use 500k for training to be responsive.
    print("Subsampling 500k rows for training to ensure timely completion...")
    if len(df) > 500000:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500000, test_size=100000, random_state=RANDOM_STATE, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        
    # Preprocessing Pipeline
    # Categorical: Target Encoding is best for high cardinality (Origin/Dest)
    # Numerical: Scaling
    
    cat_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
    num_cols = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR", "CRS_ELAPSED_TIME"]
    
    # Define Preprocessor
    # Note: TargetEncoder needs y, so we use it inside the loop or use a wrapper
    # We'll do manual encoding for simplicity and control
    print("Encoding features...")
    encoder = ce.TargetEncoder(cols=cat_cols)
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_test_enc = encoder.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = X_train_enc.copy()
    X_test_scaled = X_test_enc.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_enc[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_enc[num_cols])
    
    # Models
    models = [
        ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(class_weight="balanced", n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Extra Trees", ExtraTreesClassifier(class_weight="balanced", n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=RANDOM_STATE))
    ]
    
    if XGB_AVAILABLE:
        models.append(("XGBoost", XGBClassifier(scale_pos_weight=4, n_estimators=200, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE)))
        
    results = []
    
    # Profit Matrix (Hypothetical)
    # Cost of Delay (FN) = -$100 (Rebooking, etc)
    # Cost of Intervention (FP) = -$20 (Crew prep)
    # Benefit of Catching Delay (TP) = $50 (Net savings: Avoided $100 cost - $50 intervention cost) -> Let's say Net Benefit is +$80 vs -$100.
    # Let's simplify:
    # TN: $0
    # FP: -$20
    # FN: -$100
    # TP: $80 (Saved $100 - Spent $20)
    cost_benefit = [[0, -20], [-100, 80]]
    
    plt.figure(figsize=(10,6))
    
    for name, model in models:
        print(f"Training {name}...")
        # Use scaled data for LR, others can handle unscaled but scaled is fine
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train_enc
        X_te = X_test_scaled if name == "Logistic Regression" else X_test_enc
        
        model.fit(X_tr, y_train)
        
        # Save Model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
        
        # Predict
        y_pred = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)[:, 1]
        else:
            y_proba = model.decision_function(X_te) # Should not happen for these classifiers
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Balanced_Accuracy": bal_acc
        })
        
    # Profit Curve Calculation
    thresholds, profits = calculate_profit_curve(y_test, y_proba, cost_benefit)
    plt.plot(thresholds, profits, label=f"{name}")
    
    # Save Preprocessors
    print("Saving preprocessors...")
    preprocessors = {
        "encoder": encoder,
        "scaler": scaler,
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }
    joblib.dump(preprocessors, os.path.join(MODELS_DIR, "preprocessors.pkl"))

    # Finalize Profit Plot
    plt.title("Profit Curve Analysis")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Expected Profit (Test Set)")
    plt.legend()
    plt.axhline(0, color='black', linestyle='--')
    plt.savefig(os.path.join(PLOTS_DIR, "profit_curve.png"))
    plt.close()
    
    # Calibration Curve (Actual vs Predicted)
    plt.figure(figsize=(10,8))
    for name, model in models:
        # Load model to ensure we use the right one (or just use from memory)
        # We need probas
        X_te = X_test_scaled if name == "Logistic Regression" else X_test_enc
        y_proba = model.predict_proba(X_te)[:, 1]
        
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='.', label=name)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Actual vs Predicted)")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "calibration_curve.png"))
    plt.close()

    # Save Metrics
    metrics_df = pd.DataFrame(results)
    
    # Plot Metrics Comparison
    print("Plotting Metrics Comparison...")
    df_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="viridis")
    plt.title("Model Performance Comparison by Metric")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_all_metrics.png"))
    plt.close()

    metrics_df.to_csv(METRICS_FILE, index=False)
    print(metrics_df)
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Models saved to {MODELS_DIR}")

def main():
    df = load_data()
    run_detailed_eda(df)
    train_and_evaluate_models(df)

if __name__ == "__main__":
    main()