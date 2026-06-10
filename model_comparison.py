import numpy as np

def print_model_comparison(regression_results, classification_results):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Regression Models Comparison
    print("\nREGRESSION MODELS - R² SCORE (Higher is Better)")
    print("-" * 80)
    
    if regression_results:
        sorted_reg = sorted(regression_results, key=lambda x: x.get('r2_score', -np.inf), reverse=True)
        for model in sorted_reg:
            r2 = model.get('r2_score', np.nan)
            if isinstance(r2, (float, np.floating)) and not np.isnan(r2):
                print(f"{model['name']:30s}: {r2:.4f}")
            else:
                print(f"{model['name']:30s}: N/A")
    
    # Classification Models Comparison
    print("\n\nCLASSIFICATION MODELS - ACCURACY (Higher is Better)")
    print("-" * 80)
    
    if classification_results:
        sorted_clf = sorted(classification_results, key=lambda x: x.get('accuracy', -np.inf), reverse=True)
        for model in sorted_clf:
            acc = model.get('accuracy', np.nan)
            if isinstance(acc, (float, np.floating)) and not np.isnan(acc):
                print(f"{model['name']:30s}: {acc:.4f}")
            else:
                print(f"{model['name']:30s}: N/A")
    
    print("\n" + "=" * 80)
    
    # Best Models
    if regression_results:
        best_reg = max(regression_results, key=lambda x: x.get('r2_score', -np.inf))
        best_r2 = best_reg.get('r2_score', np.nan)
        if isinstance(best_r2, (float, np.floating)) and not np.isnan(best_r2):
            print(f"\nBest Regression Model: {best_reg['name']} with R² = {best_r2:.4f}")
    
    if classification_results:
        best_clf = max(classification_results, key=lambda x: x.get('accuracy', -np.inf))
        best_acc = best_clf.get('accuracy', np.nan)
        if isinstance(best_acc, (float, np.floating)) and not np.isnan(best_acc):
            print(f"Best Classification Model: {best_clf['name']} with Accuracy = {best_acc:.4f}")
    
    print("=" * 80)
