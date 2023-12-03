import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv('/Users/meeps360/cs158-final-project/week_approach_maskedID_timeseries.csv')

# Separate the dataset into injured and uninjured groups
injured_samples = df[df['injury'] == 1]
uninjured_samples = df[df['injury'] == 0]

# Use SMOTE to generate synthetic samples for the minority class
x = df.drop(columns=['injury', 'Athlete ID', 'Date'])
y = df['injury']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)

# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#testing oringal dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))