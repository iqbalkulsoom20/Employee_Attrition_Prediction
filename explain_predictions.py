import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Step 1: Load the cleaned dataset and trained model
data = pd.read_csv("cleaned_data.csv")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
print("Step 1: Model and data loaded successfully")

# Step 2: Preprocess the dataset
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    data[col] = data[col].astype("category").cat.codes  # Encode categorical variables

X = data.drop("Attrition", axis=1)  # Features
y = data["Attrition"]  # Target

X_scaled = scaler.transform(X)

# Step 3: Initialize LIME explainer
explainer = LimeTabularExplainer(
    X_scaled,
    training_labels=y.values,
    feature_names=X.columns,
    class_names=["No Attrition", "Attrition"],
    mode="classification",
    random_state=42,
)

# Step 4: Explain predictions for a specific instance (e.g., index 0)
instance_index = 0
explanation = explainer.explain_instance(
    X_scaled[instance_index], rf_model.predict_proba, num_features=10
)

# Step 5: Visualize explanation
print("\nExplanation for instance at index 0:")
explanation.show_in_notebook(show_table=True)
explanation.save_to_file("lime_explanation.html")
print("LIME explanation saved as 'lime_explanation.html'")

# Save explanation details to a text file
with open("lime_explanation_details.txt", "w") as f:
    f.write(str(explanation.as_list()))
print("LIME explanation details saved as 'lime_explanation_details.txt'")
