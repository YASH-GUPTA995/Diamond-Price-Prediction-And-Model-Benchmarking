import os
import sys
import pandas as pd
import importlib.util
from preprocess import preprocess_data, create_classification_target
from model_comparison import print_model_comparison

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print("=" * 80)
    print("MACHINE LEARNING MODEL EVALUATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load Data and Preprocessing
    print("\n[STEP 1] Loading and Preprocessing Data...")
    
    # Read diamonds.csv using pandas
    df = pd.read_csv("diamonds.csv")
    print(f"Loaded diamonds.csv: Shape {df.shape}")
    
    # Pass dataframe to preprocessing function
    X_train, X_test, y_train, y_test, numeric_cols = preprocess_data(df)
    y_train_clf, y_test_clf = create_classification_target(y_train, y_test)
    print("[STEP 1] Data preprocessing complete!")
    
    # Step 2: Train Regression Models
    print("\n[STEP 2] Training Regression Models...")
    regression_results = []
    
    regression_models = {
        'linear_regression': 'Linear Regression',
        'knn': 'KNN Regressor',
        'decision_tree': 'Decision Tree Regressor',
        'random_forest': 'Random Forest Regressor',
        'svm': 'SVM Regressor',
        'xgboost_model': 'XGBoost Regressor',
        'neural_network': 'MLP Regressor'
    }
    
    for module_name, model_label in regression_models.items():
        try:
            file_path = os.path.join(os.path.dirname(__file__), f'{module_name}.py')
            if os.path.exists(file_path):
                module = load_module_from_file(module_name, file_path)
                
                # Call appropriate training function
                if module_name == 'linear_regression':
                    result = module.train_linear_regression(X_train, X_test, y_train, y_test)
                elif module_name == 'knn':
                    result = module.train_knn_regressor(X_train, X_test, y_train, y_test)
                elif module_name == 'decision_tree':
                    result = module.train_decision_tree_regressor(X_train, X_test, y_train, y_test)
                elif module_name == 'random_forest':
                    result = module.train_random_forest_regressor(X_train, X_test, y_train, y_test)
                elif module_name == 'svm':
                    result = module.train_svm_regressor(X_train, X_test, y_train, y_test)
                elif module_name == 'xgboost_model':
                    result = module.train_xgboost_regressor(X_train, X_test, y_train, y_test)
                elif module_name == 'neural_network':
                    result = module.train_mlp_regressor(X_train, X_test, y_train, y_test)
                
                regression_results.append(result)
        except Exception as e:
            print(f"Error loading {model_label}: {e}")
    
    print("[STEP 2] Regression models training complete!")
    
    # Step 3: Train Classification Models
    print("\n[STEP 3] Training Classification Models...")
    classification_results = []
    
    classification_models = {
        'logistic_regression': 'Logistic Regression',
        'knn': 'KNN Classifier',
        'naive_bayes': 'Naive Bayes',
        'decision_tree': 'Decision Tree Classifier',
        'random_forest': 'Random Forest Classifier',
        'svm': 'SVM Classifier',
        'xgboost_model': 'XGBoost Classifier',
        'neural_network': 'MLP Classifier'
    }
    
    for module_name, model_label in classification_models.items():
        try:
            file_path = os.path.join(os.path.dirname(__file__), f'{module_name}.py')
            if os.path.exists(file_path):
                module = load_module_from_file(module_name, file_path)
                
                # Call appropriate training function
                if module_name == 'logistic_regression':
                    result = module.train_logistic_regression(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'knn':
                    result = module.train_knn_classifier(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'naive_bayes':
                    result = module.train_naive_bayes(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'decision_tree':
                    result = module.train_decision_tree_classifier(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'random_forest':
                    result = module.train_random_forest_classifier(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'svm':
                    result = module.train_svm_classifier(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'xgboost_model':
                    result = module.train_xgboost_classifier(X_train, X_test, y_train_clf, y_test_clf)
                elif module_name == 'neural_network':
                    result = module.train_mlp_classifier(X_train, X_test, y_train_clf, y_test_clf)
                
                classification_results.append(result)
        except Exception as e:
            print(f"Error loading {model_label}: {e}")
    
    print("[STEP 3] Classification models training complete!")
    
    # Step 4: Print Model Comparison
    print("\n[STEP 4] Generating Model Comparison Report...")
    print_model_comparison(regression_results, classification_results)
    print("[STEP 4] Model comparison complete!")
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()

