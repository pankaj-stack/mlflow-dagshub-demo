import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import dagshub


dagshub.init(repo_owner='pankaj-stack', repo_name='my-first-repo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/pankaj-stack/my-first-repo.mlflow")  # If using an MLflow server

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow Experiment
mlflow.set_experiment("DecisionTree_Iris_Experiment_dagshub") # this is the experiment name 

max_depth = 20

with mlflow.start_run():
    # Initialize and train the model
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Normalize Confusion Matrix (row-wise)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Log parameters : isme hum parameter ko log krte hain
    mlflow.log_param("max_depth", max_depth) 
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(dt, "decision_tree_model") # using sklearn it is recommended 

    # 📌 Fix: Normalize & Adjust Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names, vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")

    # Save the confusion matrix plot
    conf_matrix_path = "confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()

    # Log the artifact 
    mlflow.log_artifact(conf_matrix_path)# Log the confusion matrix 
    mlflow.log_artifact(__file__)# logging the code
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # setting the tags , useful for searching among multiple runs 
    mlflow.set_tag('author','pankaj')
    mlflow.set_tag('model','decision-tree')
    
print("🎯 Experiment Logged in MLflow!") 


# MLflow Artifacts to Track in Experiments

# ## 1️⃣ Model Artifacts
# - Trained models (e.g., RandomForest, XGBoost, PyTorch, etc.)
# - Example: `mlflow.sklearn.log_model(model, "model")`

# ## 2️⃣ Evaluation Metrics & Visualizations
# - Confusion Matrix (`confusion_matrix.png`)
# - ROC Curve (`roc_curve.png`)
# - Precision-Recall Curve (`pr_curve.png`)
# - Feature Importance (`feature_importance.png`)

# ## 3️⃣ Data & Preprocessing Artifacts
# - Dataset snapshots (`train_data.csv`, `test_data.csv`)
# - Data processing scripts (`preprocessing.py`)
# - Feature engineering pipeline (`feature_pipeline.pkl`)

# ## 4️⃣ Text Reports & Logs
# - Classification report (`classification_report.txt`)
# - Training logs (`training_log.txt`)
# - Experiment summary (`experiment_summary.txt`)

# ## 5️⃣ Hyperparameter Tuning Results
# - Best hyperparameters (`best_hyperparams.json`)
# - Grid search or Bayesian optimization results (`tuning_results.csv`)

# ## 6️⃣ Serialized Objects
# - Pickle files (`preprocessor.pkl`, `tokenizer.pkl`)
# - JSON files (`params.json`, `config.json`)

# ## 7️⃣ Custom Files
# - Any custom artifacts like PDFs, Markdown reports, or JSON files



