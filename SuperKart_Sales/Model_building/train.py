
import pandas as pd
import numpy as np
import sklearn
import mlflow
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi(token=os.getenv("SUPERKART_HF_TOKEN"))


Xtrain_path = "hf://datasets/Thaseen75/SuperKart_Sales_Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Thaseen75/SuperKart_Sales_Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Thaseen75/SuperKart_Sales_Prediction/ytrain.csv"
ytest_path = "hf://datasets/Thaseen75/SuperKart_Sales_Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


numeric_features = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP']
categorical_features = ['Product_Sugar_Content', 'Product_Type', 'Store_Type']

# 7. Preprocessing and Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

RandomForest_model = RandomForestRegressor(random_state=42)
model_pipeline = make_pipeline(preprocessor, RandomForest_model)

# 8. Hyperparameter tuning setup
parameters = {
    'randomforestregressor__n_estimators': [1000],
    'randomforestregressor__max_depth': [8],
    'randomforestregressor__min_samples_split': [10],
    'randomforestregressor__min_samples_leaf': [5],
    'randomforestregressor__max_features': [0.6],
    'randomforestregressor__max_samples': [0.7],
}

scorer = metrics.make_scorer(metrics.r2_score)

with mlflow.start_run():
    # Fit RandomizedSearchCV
    # Note: n_iter=1 because your parameter list only has one option currently
    grid_search_cv = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=parameters,
        n_iter=1, 
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        random_state=1
    )
    
    grid_search_cv.fit(X_train, y_train)

    # 9. Log CV Results (Fixed Indentation)
    results = grid_search_cv.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("cv_mean_r2", mean_score)
            mlflow.log_metric("cv_std_r2", std_score)

    # 10. Log Best Model and Evaluate
    mlflow.log_params(grid_search_cv.best_params_)
    best_model = grid_search_cv.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics_dict = {
        "train_r2": r2_score(y_train, y_pred_train),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    mlflow.log_metrics(metrics_dict)

print("Regression model logged successfully.")
print(f"Test R2 Score: {metrics_dict['test_r2']:.4f}")

# Save the model locally
model_path = "SuperKart_Sales_Prediction_v1.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "Thaseen75/SuperKart_Sales_Prediction"
repo_type = "model"

# Step 1: Check if the space exists
try:
    # Check if exists
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except Exception:
    print(f"Repo '{repo_id}' not found. Creating it...")
    # Use the string token here
    create_repo(
    repo_id=repo_id,
    repo_type=repo_type,
    space_sdk="streamlit",  # <--- Add this line
    token=os.getenv("SUPERKART_HF_TOKEN"), # Use the env variable directly
    private=False
)

# The api object already has the token, so you don't need to pass it again here
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

