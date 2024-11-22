# Breast-Cancer-Diagnosis-with-Machine-Learning-and-Deep-Learning

This notebook demonstrates a machine learning pipeline for diagnosing breast cancer using the Breast Cancer Wisconsin (Diagnostic) Dataset. The dataset includes 30 cell nucleus features derived from diagnostic imaging of 569 samples, categorized as malignant (M) or benign (B). These features capture properties like size, shape, texture, and complexity of cell nuclei, which are crucial for cancer detection. The objective is to preprocess the data, explore feature significance, and build classification models to predict diagnoses accurately. This project highlights practical applications of ML in healthcare diagnostics.

## Dataset

<img width="286" alt="dataset" src="https://github.com/user-attachments/assets/f5f3a503-96c8-421f-b641-335af4c5acc2">

## Workflow
1. Data Loading & Preprocessing: Load the dataset, handle missing values, encode categorical variables, and split into training and testing sets.
2. Model Training: Train multiple ML models such as Logistic Regression, SVM, Random Forest, Decision Tree, and K-NN.
3. Deep Learning: Build and optimize 13 deep learning models using TensorFlow and Keras.
4. Model Evaluation: Evaluate models based on accuracy, sensitivity, and specificity.
5. Model Saving & Loading: Save and load the final model for predictions.
6. Deployment: Deploy using Flaskfor inference in production.
### Flask Creation
1.Python App (app.py)
A Flask application running at http://127.0.0.1:5000/ serves the machine learning model.

2.Breast_Cancer_Detection.ipynb
Contains the code for training the machine learning model to predict breast cancer.

3.app.py
Includes Flask APIs that receive cell details through a GUI or API calls, predict the cancer diagnosis based on the trained model, and return the result.

4.templates & static Folders
Contain the HTML template and CSS styling for the user interface to input cell details and display predictions.

5.Backend Creation Using FinalModel.pkl
FinalModel.pkl
Use the pretrained FinalModel.pkl file for predictions and integrate it with the Flask application to display the output to the user.

### Web Application 
<img width="735" alt="webapp" src="https://github.com/user-attachments/assets/0e811f56-c7f3-44a5-8e83-785c23eb900b">


## Conclusion
This project successfully utilized both machine learning algorithms and advanced deep learning models for breast cancer diagnosis. Deep learning models outperformed ML approaches, achieving an impressive accuracy of 98.50%, with sensitivity and specificity metrics providing a comprehensive evaluation of their effectiveness. The final trained model was seamlessly integrated into a flask application, allowing access through API endpoints. Additionally, a user-friendly web interface was developed to enable intuitive interaction with the breast cancer diagnosis system.
