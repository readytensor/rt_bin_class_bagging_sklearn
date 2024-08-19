# Bagging Classifier in Scikit-Learn

## Project Description

This repository is a dockerized implementation of the re-usable binary classifier model. It is implemented in flexible way so that it can be used with any binary classification dataset with the use of CSV-formatted data, and a JSON-formatted data schema file. The main purpose of this repository is to provide a complete example of a machine learning model implementation that is ready for deployment.
The following are the requirements for using your data with this model:

- The data must be in CSV format.
- The number of rows must not exceed 20,000. Number of columns must not exceed 200. The model may function with larger datasets, but it has not been performance tested on larger datasets.
- Features must be one of the following two types: NUMERIC or CATEGORICAL. Other data types are not supported. Note that CATEGORICAL type includes boolean type.
- The train and test (or prediction) files must contain an ID field. The train data must also contain a target field.
- The data need not be preprocessed because the implementation already contains logic to handle missing values, categorical features, outliers, and scaling.

---

Here are the highlights of this implementation: <br/>

- A flexible preprocessing pipeline built using **SciKit-Learn** and **feature-engine**. Transformations include missing value imputation, categorical encoding, outlier removal, feature selection, and feature scaling. <br/>
- A **Bagging classifier** algorithm built using **SciKit-Learn**
- Hyperparameter-tuning using **Optuna**
  Additionally, the implementation contains the following features:
- **Data Validation**: Pydantic data validation is used for the schema, training and test files, as well as the inference request data.
- **Error handling and logging**: Python's logging module is used for logging and key functions include exception handling.

## Usage

In this section we cover the following:

- How to prepare your data for training and inference
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker
- How to use the inference service (with or without Docker)

### Preparing your data

- If you plan to run this model implementation on your own binary classification dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.

### To run locally (without Docker)

- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `requirements` directory.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`. If you want to run with hyperparameter tuning then include the `-t` flag. This will also save the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints. The service runs on port 8080.

### To run with Docker

1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t classifier_img .` <br/>
   Here `classifier_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction or inference service:
   - The train, batch predictions tasks and inference service tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete. When the inference service task is run, the container will keep running until you stop or kill it.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction or inference service tasks, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - The inference service runs on the container's port **8080**. Use the `-p` flag to map a port on local host to the port 8080 in the container.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. You can run training with or without hyperparameter tuning:
   - To run training without hyperparameter tuning (i.e. using default hyperparameters), run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train` <br/>
     where `classifier_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
   - To run training with hyperparameter tuning, issue the command: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train -t` <br/>
     This will tune hyperparameters,and used the tuned hyperparameters to train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount. It will also save the hyperparameter tuning results in the `model_inputs_outputs/outputs/hpt_outputs` directory in the bind mount.
5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.

## Requirements

The requirements files are placed in the folder `requirements`.
Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
For testing, dependencies are listed in the file `requirements_test.txt`.
Dependencies for quality-tests are listed in the file `requirements_quality.txt`. You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_test.txt
pip install -r requirements/requirements_quality.txt
```

Alternatively, you can let tox handle the installation of test dependencies for you for testing purposes. To do this, simply run the command `tox` from the root directory of the repository. This will create the environments, install dependencies, and run the tests as well as quality checks on the code.

## LICENSE

This project is provided under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)
