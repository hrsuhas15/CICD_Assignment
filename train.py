import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Define Random Forest Classifier with AdaBoost
model = AdaBoostClassifier()

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0, 10.0]
}

# Perform feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Get the best model from the grid search
model = grid_search.best_estimator_

# Train the best model on the entire training set
model.fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
