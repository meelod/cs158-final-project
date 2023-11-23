import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming your dataset is in a CSV file
df = pd.read_csv('your_dataset.csv')

# Example: Handle missing values by filling with the mean
df = df.fillna(df.mean())

# Example: Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df)

X = df.drop('target_column', axis=1)  # Features
y = df['target_column']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Other evaluation metrics
print(classification_report(y_test, y_pred))
