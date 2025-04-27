import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Step 1: Load your dataset
df = pd.read_csv("filtered_data.csv")  # Replace with your file path

# Step 2: Separate features and target
X = df.drop(columns=["readmitted"]).values
y = df["readmitted"].values

# Step 3: Compute mutual information scores
mi_scores = mutual_info_classif(X, y, discrete_features='auto')

# Step 4: Select top 4 features with highest MI
top_k = 4
top_feature_indices = np.argsort(mi_scores)[-top_k:]
X_top_features = X[:, top_feature_indices]

# Step 5: Standardize selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top_features)

# Step 6: Convert to DataFrame and add labels
df_features = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(top_k)])
df_features["label"] = y

# Step 7: Separate by class
class_0 = df_features[df_features["label"] == 0].copy()
class_1 = df_features[df_features["label"] == 1].copy()

# Step 8: Compute L2 norm for each row in top MI features
class_0["score"] = np.linalg.norm(class_0.iloc[:, :top_k].values, axis=1)
class_1["score"] = np.linalg.norm(class_1.iloc[:, :top_k].values, axis=1)

# Step 9: Select top 2500 extreme samples from each class
top_2500_0 = class_0.sort_values(by="score", ascending=False).head(2500)
top_2500_1 = class_1.sort_values(by="score", ascending=False).head(2500)

# Step 10: Get original rows corresponding to selected indices
selected_indices = top_2500_0.index.tolist() + top_2500_1.index.tolist()
balanced_subset = df.iloc[selected_indices]

# Step 11: Save the final balanced subset
balanced_subset.to_csv("top_5000_mi_balanced_subset.csv", index=False)
print("Saved balanced subset to top_5000_mi_balanced_subset.csv")
