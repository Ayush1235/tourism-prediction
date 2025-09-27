import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------
data_path = "tourism_project/data/tourism.csv"  # adjust path if needed
df = pd.read_csv(data_path)
print("Dataset loaded successfully. Shape:", df.shape)

# -----------------------------------------------------
# 2. Drop unnecessary columns
# -----------------------------------------------------
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True, errors='ignore')

# Strip whitespace and standardize case
df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
df['MaritalStatus'] = df['MaritalStatus'].astype(str).str.strip().str.title()

# Apply normalization
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# -----------------------------------------------------
# 3. Label encode categorical columns
# -----------------------------------------------------
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("Label encoding applied to:", categorical_cols)

# -----------------------------------------------------
# 4. Train–test split
# -----------------------------------------------------
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Split complete. Train size:", Xtrain.shape[0], "Test size:", Xtest.shape[0])

# -----------------------------------------------------
# 5. Handle class imbalance (scale_pos_weight)
# -----------------------------------------------------
neg, pos = ytrain.value_counts()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# -----------------------------------------------------
# 6. Define model and hyperparameter grid
# -----------------------------------------------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    random_state=42
)

param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.5, 0.7],
    'subsample': [0.8, 1.0]
}

# -----------------------------------------------------
# 7. Train with GridSearchCV and MLflow logging
# -----------------------------------------------------
with mlflow.start_run():
    grid = GridSearchCV(
        xgb_model,
        param_grid,
        scoring='recall',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(Xtrain, ytrain)

    mlflow.log_params(grid.best_params_)
    best_model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)

    # Evaluate
    y_pred_test = best_model.predict(Xtest)
    print("Classification Report (Test):\n", classification_report(ytest, y_pred_test))

    # Log metrics
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_recall": test_report['1']['recall'],
        "test_precision": test_report['1']['precision'],
        "test_f1": test_report['1']['f1-score']
    })

    # Save artifacts locally
    os.makedirs("tourism_project/models", exist_ok=True)
    model_path = "tourism_project/models/xgb_tourism_model.pkl"
    joblib.dump(best_model, model_path)
    print("Model saved locally:", model_path)

    # Save feature names
    feature_names = Xtrain.columns.tolist()
    feature_names_path = "tourism_project/models/feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)

    # Save encoders (optional but useful for inference)
    encoders_path = "tourism_project/models/label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)

# -----------------------------------------------------
# 8. Upload artifacts to Hugging Face Hub
# -----------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
repo_id = "ayush12358/tourism-xgb-model2"

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Hugging Face repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Created new Hugging Face model repo: {repo_id}")

# Upload model and preprocessing artifacts
for file_path in [model_path, feature_names_path, encoders_path]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Uploaded {os.path.basename(file_path)} to Hugging Face repo.")
