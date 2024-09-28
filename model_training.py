from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

def balance_classes(X, y):
    """Balance classes using SMOTE."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def tune_hyperparameters(model_name, model, X_train, y_train):
    """Tune hyperparameters for the given model using GridSearchCV."""
    
    if model_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
    elif model_name == "XGBoost":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8],
            'classifier__colsample_bytree': [0.8]
        }
        
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: ", grid_search.best_params_)
    
    return grid_search.best_estimator_
