import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv('/Users/meeps360/cs158-final-project/week_approach_maskedID_timeseries.csv')

# Separate the dataset into injured and uninjured groups
injured_samples = df[df['injury'] == 1]
uninjured_samples = df[df['injury'] == 0]

# Randomly sample half of the uninjured group
uninjured_samples_subset = uninjured_samples.sample(n=len(injured_samples), random_state=42)

# Concatenate the injured and uninjured subsets
balanced_df = pd.concat([injured_samples, uninjured_samples_subset])

# Shuffle the rows to randomize the order
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Now, balanced_df is a DataFrame with equal representation of both classes

# Drop unnecessary columns
df = balanced_df.drop(columns=['Athlete ID', 'Date'])

# Split the data
X = balanced_df.drop(columns=['injury'])
y = balanced_df['injury']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Model Training with Grid Search
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     # Add other parameters to be tuned
# }

# clf = RandomForestClassifier(random_state=42)

# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")

# Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))