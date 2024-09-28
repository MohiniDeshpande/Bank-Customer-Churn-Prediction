import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path, skipinitialspace=True)
        data.columns = data.columns.str.strip()  # Strip trailing spaces from column names
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the dataset by encoding categorical variables and handling missing values."""
    print("Initial Columns:", data.columns.tolist())

    # Handle categorical variables
    categorical_cols = ['marital', 'housing', 'loan', 'default', 'education', 'job', 'poutcome', 'contact', 'month']

    # Encode categorical variables
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')  # Convert to categorical dtype

    # Convert categorical to numerical using one-hot encoding
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Replace 'pdays' 999 with 0
    if 'pdays' in data.columns:
        data['pdays'] = data['pdays'].replace(999, 0)
    
    # Ensure 'y' column is present before dropping it
    if 'y' in data.columns:
        target = data['y'].str.strip()  # Strip whitespace from target variable
        target = target.map({'no': 0, 'yes': 1})  # Convert to numeric
        data.drop(['y'], axis=1, inplace=True)
    else:
        raise KeyError("Target variable 'y' not found in DataFrame after preprocessing.")
    
    return data, target
