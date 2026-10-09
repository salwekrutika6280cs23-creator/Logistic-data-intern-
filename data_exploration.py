# ============================================================
# Logistics Data Analysis - Week 1
# Strategic Planning and Data Exploration
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 1. Project Configuration
# ------------------------------------------------------------

DATA_PATH = "data/dynamic_supply_chain_logistics_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

print("=" * 70)
print("LOGISTICS DATA ANALYSIS - WEEK 1")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ------------------------------------------------------------
# 3. Basic Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumn Names:")
for column in df.columns:
    print("-", column)


# ------------------------------------------------------------
# 4. Data Types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 5. Missing Value Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")


# ------------------------------------------------------------
# 6. Duplicate Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate records: {duplicate_count}")


# ------------------------------------------------------------
# 7. Convert Timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print("\nTimestamp converted successfully.")


# ------------------------------------------------------------
# 8. Create Time-Based Features
# ------------------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["date"] = df["timestamp"].dt.date

print("Time-based features created.")


# ------------------------------------------------------------
# 9. Statistical Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

summary = df.describe()

print(summary)


# Save summary
summary.to_csv(
    os.path.join(OUTPUT_DIR, "statistical_summary.csv")
)


# ------------------------------------------------------------
# 10. Risk Classification Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK CLASSIFICATION")
print("=" * 70)

risk_counts = df["risk_classification"].value_counts()

print(risk_counts)

risk_counts.to_csv(
    os.path.join(OUTPUT_DIR, "risk_classification_counts.csv")
)


# ------------------------------------------------------------
# 11. KPI 1 - Average Delivery Delay
# ------------------------------------------------------------

average_delay = df["delivery_time_deviation"].mean()

print("\n" + "=" * 70)
print("KPI ANALYSIS")
print("=" * 70)

print(
    f"Average Delivery Time Deviation: "
    f"{average_delay:.2f} hours"
)


# ------------------------------------------------------------
# 12. KPI 2 - Average Shipping Cost
# ------------------------------------------------------------

average_shipping_cost = df["shipping_costs"].mean()

print(
    f"Average Shipping Cost: "
    f"{average_shipping_cost:.2f}"
)


# ------------------------------------------------------------
# 13. KPI 3 - Average Lead Time
# ------------------------------------------------------------

average_lead_time = df["lead_time_days"].mean()

print(
    f"Average Lead Time: "
    f"{average_lead_time:.2f} days"
)


# ------------------------------------------------------------
# 14. KPI 4 - Average Fuel Consumption
# ------------------------------------------------------------

average_fuel = df["fuel_consumption_rate"].mean()

print(
    f"Average Fuel Consumption Rate: "
    f"{average_fuel:.2f}"
)


# ------------------------------------------------------------
# 15. KPI 5 - Average Delay Probability
# ------------------------------------------------------------

average_delay_probability = df["delay_probability"].mean()

print(
    f"Average Delay Probability: "
    f"{average_delay_probability:.4f}"
)


# ------------------------------------------------------------
# 16. KPI Summary Table
# ------------------------------------------------------------

kpi_summary = pd.DataFrame({
    "KPI": [
        "Average Delivery Time Deviation",
        "Average Shipping Cost",
        "Average Lead Time",
        "Average Fuel Consumption Rate",
        "Average Delay Probability"
    ],
    "Value": [
        average_delay,
        average_shipping_cost,
        average_lead_time,
        average_fuel,
        average_delay_probability
    ],
    "Unit": [
        "Hours",
        "Cost Units",
        "Days",
        "Fuel Units",
        "Probability"
    ]
})

print("\nKPI Summary:")
print(kpi_summary)

kpi_summary.to_csv(
    os.path.join(OUTPUT_DIR, "kpi_summary.csv"),
    index=False
)


# ------------------------------------------------------------
# 17. Traffic vs Delivery Delay
# ------------------------------------------------------------
# Convert continuous traffic values into practical levels
df["traffic_level"] = (
    df["traffic_congestion_level"]
    .round()
    .clip(0, 10)
    .astype(int)
)

traffic_delay = df.groupby(
    "traffic_level"
)["delivery_time_deviation"].mean()

print("\n" + "=" * 70)
print("TRAFFIC CONGESTION VS DELIVERY DELAY")
print("=" * 70)

print(traffic_delay)


# ------------------------------------------------------------
# 18. Loading Time vs Delivery Delay
# ------------------------------------------------------------

loading_delay = df[
    [
        "loading_unloading_time",
        "delivery_time_deviation"
    ]
].corr()

print("\n" + "=" * 70)
print("LOADING TIME VS DELIVERY DELAY CORRELATION")
print("=" * 70)

print(loading_delay)


# ------------------------------------------------------------
# 19. Correlation Analysis
# ------------------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_df.corr()

correlation_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)

print("\nCorrelation matrix saved.")


# ------------------------------------------------------------
# 20. Visualization - Risk Classification
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="risk_classification"
)

plt.title("Risk Classification Distribution")
plt.xlabel("Risk Classification")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "risk_classification_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 21. Visualization - Traffic vs Delay
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="traffic_level",
    y="delivery_time_deviation"
)

plt.title("Traffic Congestion Level vs Delivery Time Deviation")
plt.xlabel("Traffic Congestion Level (0-10)")
plt.ylabel("Delivery Time Deviation (Hours)")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "traffic_vs_delivery_delay.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 22. Visualization - Loading Time vs Delay
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="loading_unloading_time",
    y="delivery_time_deviation",
    alpha=0.5
)

plt.title("Loading/Unloading Time vs Delivery Time Deviation")
plt.xlabel("Loading/Unloading Time")
plt.ylabel("Delivery Time Deviation (Hours)")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loading_time_vs_delay.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 23. Visualization - Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Logistics Dataset Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 24. Hourly Delivery Delay Analysis
# ------------------------------------------------------------

hourly_delay = df.groupby(
    "hour"
)["delivery_time_deviation"].mean()

plt.figure(figsize=(10, 5))

hourly_delay.plot(
    marker="o"
)

plt.title("Average Delivery Time Deviation by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delivery Time Deviation (Hours)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hourly_delivery_delay.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 25. Save Cleaned Dataset
# ------------------------------------------------------------

cleaned_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_logistics_dataset.csv"
)

df.to_csv(
    cleaned_path,
    index=False
)

print("\nCleaned dataset saved.")


# ------------------------------------------------------------
# 26. Final Output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(f"\nOutput files are available inside: {OUTPUT_DIR}/")

print("\nGenerated files:")

for file in os.listdir(OUTPUT_DIR):
    print("-", file)

print("\nProject Week 1 analysis completed.")