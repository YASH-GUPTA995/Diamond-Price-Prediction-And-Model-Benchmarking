import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    print("Dropped duplicates")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Replace 0 with NaN for numeric columns
    df[numeric_cols] = df[numeric_cols].replace(0, np.nan)
    
    # Fill missing values (avoid chained assignment / inplace)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("\nMissing values after filling:\n", df.isnull().sum())
    
    # Encode categorical variables
    label_enc = LabelEncoder()
    for col in cat_cols:
        df[col] = label_enc.fit_transform(df[col])
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    print("\nShape after outlier removal:", df.shape)
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Split into features and target
    target = "price"
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nFinal Train/Test Shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    return X_train, X_test, y_train, y_test, numeric_cols

def create_classification_target(y_train, y_test):
    y_train_clf = pd.cut(y_train, bins=3, labels=[0, 1, 2])
    y_test_clf = pd.cut(y_test, bins=3, labels=[0, 1, 2])
    return y_train_clf, y_test_clf
