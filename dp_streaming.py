import joblib
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col

# ==========================================
# Developer: Anthony Artola
# Task: AI-Driven Adaptive Noise Injection
# ==========================================

spark = SparkSession.builder.appName("Adaptive_Security_Anthony_Artola").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load the "Blurry Brain"
model = joblib.load('dp_income_model.pkl')

lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
parsed = lines.select(
    split(col("value"), ",").getItem(0).cast("int").alias("age"),
    split(col("value"), ",").getItem(1).cast("int").alias("capital_gain")
)

def adaptive_process(batch_df, batch_id):
    if batch_df.count() > 0:
        print(f"\nANALYTIC BATCH ID: {batch_id} {'='*30}")
        data = np.array(batch_df.select("age", "capital_gain").collect())
        
        for row in data:
            if row[0] is None: continue
            
            # 1. AI PREDICTION
            prediction = model.predict([row])[0]
            label = ">50K" if prediction == 1 else "<=50K"
            
            # 2. CALCULATE CONFIDENCE (The "AI Intuition")
            probs = model.predict_proba([row])[0]
            confidence = max(probs)
            
            # 3. GENERATE ADAPTIVE NOISE
            # If confidence is low (0.5), noise scale is higher. 
            # If confidence is high (1.0), noise scale is lower.
            noise_scale = (1.0 - confidence) * 10 
            noise_added = np.random.laplace(0, noise_scale)
            
            # 4. OUTPUT RESULTS
            print(f"INPUT RECEIVED  | Age: {row[0]}, Gain: ${row[1]}")
            print(f"AI CONFIDENCE   | {confidence*100:.2f}%")
            print(f"ADAPTIVE NOISE  | {noise_added:.4f} (Based on uncertainty)")
            print(f"FINAL RESULT    | {label}")
            print(f"{'-'*45}")

query = parsed.writeStream.foreachBatch(adaptive_process).start()
query.awaitTermination()