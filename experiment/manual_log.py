# Importing dependencies
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import sys

# Setting an experiment for manual logging
mlflow.set_experiment("manual_logging")

# Function for manual logging
if __name__ == "__main__":     
    # Loading data
    data = datasets.load_breast_cancer()
    
    # Splitting the data into traing and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                        data.target,
                                                        stratify=data.target)
    
    # Selecting a parameter range to try out
    C = list(range(1, 5))     
    
    # Starting a tracking run
    with mlflow.start_run(run_name="PARENT_RUN"):
        # For each value of C, run a child run
        for value in C:
            with mlflow.start_run(run_name="CHILD_RUN", nested=True):
                # Instantiating and fitting the model
                model = LogisticRegression(C=value, max_iter=int(sys.argv[1]))
                model.fit(X_train, y_train)
                
                # Logging the current value of C
                mlflow.log_param("C", value)
                
                # Logging the test performance of the current model                
                mlflow.log_metric("Score", value=model.score(X_test, y_test)) 
                
                # Saving the model as an artifact
                mlflow.sklearn.log_model(model, "model")