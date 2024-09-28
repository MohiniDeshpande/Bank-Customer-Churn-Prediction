This project aims to predict customer churn in the banking sector using various machine learning algorithms. By analyzing historical customer data, we can identify factors that contribute to customer retention or loss. The goal is to build an effective predictive model that assists banks in implementing strategies to improve customer satisfaction and reduce churn rates.

**Features:**

Data Loading and Preprocessing: Load and clean the dataset, handle missing values, and encode categorical variables to prepare the data for analysis.
Class Imbalance Handling: Utilize the SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the target variable.

Multiple Machine Learning Algorithms: Implement and evaluate several classification models, including;
Decision Tree, Random Forest, Support Vector Machine (SVM), XGBoost, Logistic Regression

Hyperparameter Tuning: Optimize the performance of Random Forest and XGBoost models through grid search for hyperparameter tuning.

Model Evaluation: Assess model performance using metrics such as accuracy, AUC-ROC, and confusion matrix visualization.

Model Comparison Visualization: Compare the performance of different models using bar plots for clarity.

Model Persistence: Save the best-performing model for future use, ensuring that it can be easily deployed.

**Files:**

1> main.py: The primary script that orchestrates the workflow, including data loading, preprocessing, model training, evaluation, and saving the best model.

2> requirements.txt: Lists the required Python packages to run the project, including pandas, numpy, scikit-learn, xgboost, and imbalanced-learn.

3> bank.csv: The dataset used for training and evaluating the churn prediction models.

**Requirements**
To run this project, you will need the following Python packages:
pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, imbalanced-learn, joblib.

You can install the required packages using:

pip install -r requirements.txt

**How to Run**

1> Clone the repository:

git clone <repository-url>
cd <repository-directory>

2> Place your dataset (bank.csv) in the root directory.

3>Run the main script:

python main.py


Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or any issues, feel free to open an issue or submit a pull request.
