# USE THIS SCRIPT TO RUN THE PROJECT FROM PYTHON

import mlflow

mlflow.projects.run(uri="experiment",
                    entry_point="manual_logger",
                    parameters={"max_iter": 10000},
                    experiment_name="manual_logging")