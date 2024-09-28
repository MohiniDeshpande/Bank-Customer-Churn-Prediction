from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, auc_roc

def visualize_results(results):
    """Visualize performance comparison of models."""
    results_df = pd.DataFrame(results).T

    results_df[['Accuracy', 'AUC-ROC']].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.show()
