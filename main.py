import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Import custom modules
from data_processing import load_data, preprocess_data
from model_training import balance_classes, tune_hyperparameters
from evaluate_models import evaluate_model, visualize_results

# Configure logging
logging.basicConfig(level=logging.INFO)

def main(file_path):
    """Main function to execute the workflow."""
    
    # Load and preprocess the data
    data = load_data(file_path)
    
    logging.info("Initial Data Shape: %s", data.shape)

    # Preprocess the data and separate features and target variable
    data, target = preprocess_data(data)
    
    logging.info("Processed Data Shape: %s", data.shape)

    X = data
    y = target

    # Balance classes and split the dataset into train and test sets
    X_balanced, y_balanced = balance_classes(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

   # Initialize models for comparison 
     models = {
       "Decision Tree": DecisionTreeClassifier(),
       "Random Forest": RandomForestClassifier(),
       "SVM": svm.SVC(probability=True),
       "XGBoost": XGBClassifier(use_label_encoder=False),
       "Logistic Regression": LogisticRegression(max_iter=1000)
   }

    results = {}

   # Evaluate each model 
    for name, model in models.items():
       print(f"Training {name}...")
       
       if name in ["Random Forest", "XGBoost"]:
           model_pipeline = Pipeline([
               ('scaler', StandardScaler()),
               ('classifier', model)
           ])
           best_model = tune_hyperparameters(name, model_pipeline, X_train, y_train)
           accuracy, auc_roc = evaluate_model(best_model, X_test, y_test)
           results[name] = {"Accuracy": accuracy, "AUC-ROC": auc_roc}
           print(f"{name} (Tuned) - Accuracy: {accuracy:.2f}%, AUC-ROC: {auc_roc:.2f}")
       else:
           model.fit(X_train, y_train)
           accuracy, auc_roc = evaluate_model(model, X_test, y_test)
           results[name] = {"Accuracy": accuracy, "AUC-ROC": auc_roc}
           print(f"{name} - Accuracy: {accuracy:.2f}%, AUC-ROC: {auc_roc:.2f}")

    visualize_results(results)

   # Best model based on AUC-ROC score 
    best_model_name = max(results.items(), key=lambda x: x[1]['AUC-ROC'])[0]
    print(f"The best model is {best_model_name} with an AUC-ROC of {results[best_model_name]['AUC-ROC']:.2f}.")

   # Save the best model to disk 
    best_model_instance = models[best_model_name]
    joblib.dump(best_model_instance, f'{best_model_name.replace(" ", "_").lower()}_model.pkl')
    print(f"Model saved as '{best_model_name.replace(' ', '_').lower()}_model.pkl'.")

# Execute main function when script is run directly 
if __name__ == "__main__":
   main('/bank.csv')
