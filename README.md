# Credit Card Approval ML Model Evaluation API - Data Team

### Project Overview

This repository contains the API for predicting credit card approval status using a machine learning model. The model is trained using logistic regression, optimized for performance, and integrated into a Flask-based API.

### Key Features
- **Model Training**: Building and training a classification model using machine learning techniques.
- **Model Considerations**: Addressing key concerns such as bias, variance, fairness, and interpretability while training the model.
- **API Development**: Developing an API to serve the trained model, enabling predictions based on input features.

### Directory Structure
```
├── LICENSE
├── README.md
├── MLEvaluation_API.postman_collection.json   # Postman collection for testing the API
├── optimized_logistic_regression_model.pkl    # Pre-trained logistic regression model
├── requirements.txt   # Dependencies required to run the project
├── app.py   # Flask application hosting the API
├── MLEvaluation_Task.ipynb  # Notebook including the model work
├── credit_card_train.csv   # Data used for training the model  
├── Task Description.pdf  # Task description
```

### Requirements
To install and run the project, ensure you have the following dependencies:
- **Python 3.x**
- **Flask** - Micro web framework for serving the API
- **Machine Learning Libraries**:
  - pandas

These dependencies are included in the `requirements.txt` file. You can install them by running:

```bash
pip install -r requirements.txt
```

### Running the Application

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure Python 3.x is installed. Use the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App**:
   After installing dependencies, run the app:
   ```bash
   python app.py
   ```
   By default, the server will run on `http://127.0.0.1:5000/`.

### API Usage

- **Endpoint**: `POST /predict`
- **Description**: Make a credit card approval prediction based on input features.

#### Request Body Example:
```json
{
  "Num_Children": [1, 2, 1, 1, 1, 4, 2, 4, 4, 3, 3, 0, 5, 3, 1],
  "Gender": ["Male", "Female", "Male", "Male", "Male", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Male", "Male"],
  "Income": [40690, 75469, 70497, 61000, 56666, 88940, 76331, 98610, 74190, 65759, 99948, 66068, 65982, 72445, 119407],
  "Own_Car": ["No", "Yes", "Yes", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes"],
  "Own_Housing": ["Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No"],
  "Credit_Card_Issuing": ["Denied", "Denied", "Approved", "Denied", "Denied", "Approved", "Denied", "Approved", "Denied", "Denied", "Approved", "Denied", "Denied", "Denied", "Approved"]
}
```

#### Response Body Example:
```json
{
  "predictions": [
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    1
  ]
}
```
Where `1` indicates credit card approval and `0` indicates denial.

### Important Notes
- The model is designed with considerations for **performance, bias, fairness**, and **interpretability**.
- For testing, use the Postman collection file `MLEvaluation_API.postman_collection.json` provided in the repository.

### License
This project is licensed under the Apache-02 License. See the [LICENSE](LICENSE) file for details.

### Contact
For any issues or questions, feel free to reach out via Teams or email.
