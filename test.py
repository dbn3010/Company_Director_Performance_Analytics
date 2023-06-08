import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

# Load the dataset
dataset_path = r"C:\Users\durga\OneDrive\Documents\test\lending_club_loan_dataset.csv"
df = pd.read_csv(dataset_path)

# Select relevant features and target variable
features = ['grade', 'annual_inc', 'emp_length_num', 'dti', 'home_ownership', 'purpose', 'term',
            'last_delinq_none', 'revol_util', 'total_rec_late_fee']
target = 'bad_loan'

# Preprocess the data
df = df[features + [target]].dropna()  # Drop rows with missing values
X = df[features]
y = df[target]

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = label_encoder.fit_transform(X[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
