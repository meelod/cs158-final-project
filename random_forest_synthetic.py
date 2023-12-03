import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv('/Users/meeps360/cs158-final-project/week_approach_maskedID_timeseries.csv')

# Separate the dataset into injured and uninjured groups
injured_samples = df[df['injury'] == 1]
uninjured_samples = df[df['injury'] == 0]

# Use SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


#notes:
# When i first started writing this code and have what i have above
# there is an issue where the model is not predicting the injuries. My guess
# is because there is not enough data that show when an injury occurs.

# Now I will try to reduce the data so that half is injuries and half is not injuries

# after doing that, i am now getting an accuracy of 66% that predicts if a runner is going to be injured or not

# now i will try creating synthetic data or duplicates so that we do not remove most of the data