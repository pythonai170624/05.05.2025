import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# 1. Create clean artificial dataset with 3 clear classes
X = np.array([
    [200, 7, 150], [210, 7, 160], [190, 6, 140], [180, 6, 145], [220, 8, 155],     # apples
    [240, 9, 170], [250, 8, 165], [230, 9, 180], [235, 9, 175], [245, 8, 169],     # oranges
    [30, 12, 120], [40, 13, 130], [20, 11, 115], [35, 13, 125], [25, 12, 118],     # bananas
    [195, 8, 158], [225, 9, 160], [38, 10, 110], [245, 7, 172], [210, 6, 148]      # noisy/mixed
])

y = np.array([
    'apple', 'apple', 'apple', 'apple', 'apple',
    'orange', 'orange', 'orange', 'orange', 'orange',
    'banana', 'banana', 'banana', 'banana', 'banana',
    'apple', 'orange', 'banana', 'orange', 'apple'
])


# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale data (important for KNN)
# for train and split

# 4. Train basic KNN

# 5. Predict and evaluate

print("=== Initial Evaluation ===")
print("Accuracy:", )
print("Classification Report:\n", )
print("Confusion Matrix:\n", )

# 6. GridSearchCV to find best k
param_grid = {
    'n_neighbors': [1, 3, 5],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan or Euclidean
}

# GridSearchCV with param_grid, cv=3, scoring='accuracy'

# get the best_knn model from grid.best_estimator_ 

# 7. Predict and evaluate again

print("\n=== After GridSearchCV ===")
print("Best parameters:", )
print("Best CV score:", )
print("Accuracy:", )
print("Classification Report:\n", )
print("Confusion Matrix:\n", )
