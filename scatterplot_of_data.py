import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv(r"8-fraud_detection.csv")

X = df["transaction_amount"]
y = df["transaction_risk_score"]

plt.figure(figsize=(10, 6), dpi=100)
sns.scatterplot(x=X, y=y, hue=df["is_fraud"], alpha=0.7, s = 100)
plt.xlabel("Transaction Amount", fontsize=15)
plt.ylabel("Transaction Risk Score", fontsize=15)
plt.title("Scatter Plot of Transaction Data", fontsize=20, fontweight = "bold")
plt.legend(title="is_fraud", fontsize=14, title_fontsize=16)
plt.grid()
plt.tight_layout()

plt.show()
