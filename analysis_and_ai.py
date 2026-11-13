import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pyspark.sql import SparkSession
from diffprivlib.models import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Developer: Anthony Artola
# Project: Big Data Analytics & Security
# Task: DP Model Training & Performance Benchmarking
# ==========================================

# Initialize Spark for the session (Optional for the AI math, but good for the environment)
spark = SparkSession.builder \
    .appName("DP_Training_Anthony_Artola") \
    .getOrCreate()

# 1. DATA ACQUISITION: Loading the US Adult Census Benchmark
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
        "hours-per-week", "native-country", "income"]

# Load and Clean (Handling the '?' missing values common in Census data)
df = pd.read_csv(data_url, names=cols, na_values=" ?", skipinitialspace=True).dropna()

# 2. FEATURE ENGINEERING: Selecting Age and Capital Gain as our signal
# We use LabelEncoder to turn the string "Income" into 0s and 1s
df_features = df[['age', 'capital-gain']].copy()
df_target = LabelEncoder().fit_transform(df['income'])

# Split the data: 80% for the "Blurry Study", 20% for the "Blind Test"
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2)

# 3. PRIVACY BENCHMARKING: Testing the Utility-Privacy Trade-off
epsilons = np.logspace(-3, 1, 15)
accuracies = []

print("Starting Privacy-Utility Analysis...")
for eps in epsilons:
    # bounds=([min_age, min_gain], [max_age, max_gain]) sets the sensitivity
    model = GaussianNB(epsilon=eps, bounds=([17, 0], [90, 100000]))
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))
    print(f"Tested Epsilon: {eps:.4f} | Accuracy: {accuracies[-1]:.4f}")

# 4. EXPORTING THE "BRAIN": Saving the optimal model for the Stream
# We choose Epsilon 1.0 as the "Sweet Spot" for the live demo
print("\nFinalizing optimal model (Epsilon=1.0) for deployment...")
final_model = GaussianNB(epsilon=1.0, bounds=([17, 0], [90, 100000]))
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'dp_income_model.pkl')
print("FILE EXPORTED: dp_income_model.pkl is ready for streaming_dp.py")

# 5. VISUALIZATION: Generating the Research Evidence
plt.figure(figsize=(10, 6))
plt.semilogx(epsilons, accuracies, marker='o', color='teal', linewidth=2)
plt.title("The Privacy-Utility Frontier (US Adult Census)")
plt.xlabel("Privacy Budget (Epsilon ε)")
plt.ylabel("AI Model Accuracy")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.axvline(x=1.0, color='red', linestyle='--', label='Deployment Epsilon (1.0)')
plt.legend()
plt.show()

spark.stop()