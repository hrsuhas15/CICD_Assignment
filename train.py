import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Define preprocessing steps (scaling numerical features)
preprocessor = StandardScaler()

# Define models to try
models_to_try = [
    ('RandomForest', RandomForestClassifier()),
    ('GradientBoosting', GradientBoostingClassifier())
]
    
best_accuracy = 0
best_model = None

for model_name, model in models_to_try:
    # Create pipeline with preprocessing and modeling steps
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    # Get the best model from the grid search
    model = grid_search.best_estimator_

    # Train the best model on the entire training set
    model.fit(X, y)

    # Evaluate model performance
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Save the best model if it improves accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

model = best_model

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
