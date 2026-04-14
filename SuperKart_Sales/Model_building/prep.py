
# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("SUPERKART_HF_TOKEN"))
DATASET_PATH = "hf://datasets/Thaseen75/SuperKart_Sales_Prediction/SuperKart.csv"
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

def treat_outliers(df,cols):
	Q1 = df[cols].quantile(0.25)
	Q3 = df[cols].quantile(0.75)
	IQR = Q3 - Q1
	lower_whisker = Q1 - 1.5 * IQR
	upper_whisker = Q3 + 1.5 * IQR

	df[cols] = np.clip(df[cols],lower_whisker,upper_whisker,axis=1)
	return df

data = treat_outliers(data,numerical_features)

# Drop the unique identifier
cols_to_drop = [
    'Product_Id', 
    'Product_Store_Sales_Total', 
    'Store_Establishment_Year', 
    'Store_Id', 
    'Store_Location_City_Type', 
    'Store_Size'
]

# 3. Define X and y FIRST
X = data.drop(cols_to_drop, axis=1)
y = data['Product_Store_Sales_Total']

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Thaseen75/SuperKart_Sales_Prediction",
        repo_type="dataset",
    )
