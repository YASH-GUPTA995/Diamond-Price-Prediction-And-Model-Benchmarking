import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

def train_knn_regressor(X_train, X_test, y_train, y_test):
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train, y_train)
    y_pred = knn_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 60)
    print("K-NEAREST NEIGHBORS (KNN) - REGRESSION")
    print("=" * 60)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 60)
    
    return {
        'name': 'KNN Regressor',
        'type': 'regression',
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def train_knn_classifier(X_train, X_test, y_train_clf, y_test_clf):
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train_clf)
    y_pred = knn_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test_clf, y_pred)
    precision = precision_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS (KNN) - CLASSIFICATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("=" * 60)
    
    return {
        'name': 'KNN Classifier',
        'type': 'classification',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
