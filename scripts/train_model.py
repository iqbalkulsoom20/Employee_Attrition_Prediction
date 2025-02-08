# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Step 1: Load the cleaned dataset
data = pd.read_csv("cleaned_data.csv")
print("Step 1: Dataset loaded successfully")

# Step 2: Preprocessing
# Identify and encode categorical columns
categorical_cols = data.select_dtypes(include=["object"]).columns
print(f"Categorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data.drop("Attrition", axis=1)  # Features (drop target column)
y = data["Attrition"]  # Target (Attrition column)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"Train-test split completed. Training size: {len(X_train)}, Test size: {len(X_test)}")

# Step 4: Train and evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully")

rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.2f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# Save the Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved successfully")

