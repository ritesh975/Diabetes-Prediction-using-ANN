# 🩺 Diabetes Prediction using ANN

An interactive **Diabetes Prediction Web Application** built using **Artificial Neural Network (ANN)** with **Sigmoid Activation** and deployed using **Streamlit**.

The application takes patient health information as input and predicts the probability of diabetes using a trained neural network model.

---

## 🚀 Live Demo

🔗 **Demo:** (https://ritesh-diabetes.streamlit.app/)

---

## 📂 GitHub Repository

🔗 **Project Repository:** (https://github.com/ritesh975/Diabetes-Prediction-using-ANN)

---

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** to perform binary classification for diabetes prediction.

The model is trained on patient health-related features and predicts whether a person is likely to have diabetes.

The trained ANN model is connected to a **Streamlit UI**, allowing users to enter patient information and receive a real-time prediction.

---

## ✨ Features

* 🧠 Artificial Neural Network (ANN)
* 🔵 Sigmoid Activation Function
* 📊 Binary Classification
* 🩺 Diabetes Prediction
* 📈 Probability-based prediction
* 🖥️ Interactive Streamlit UI
* 🔄 Real-time prediction
* 📋 Patient input form
* 📊 Prediction probability display
* 💾 Trained model saved for deployment
* ⚙️ Feature scaling using StandardScaler

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* TensorFlow
* Keras
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Model

* Artificial Neural Network (ANN)
* Sigmoid Activation
* Binary Crossentropy
* Adam Optimizer

### Deployment / UI

* Streamlit

### Model Saving

* Joblib
* Keras `.keras` model format

---

## 🧠 ANN Architecture

```text
Input Features
      │
      ▼
Dense Layer
16 Neurons
Sigmoid Activation
      │
      ▼
Dense Layer
8 Neurons
Sigmoid Activation
      │
      ▼
Output Layer
1 Neuron
Sigmoid Activation
      │
      ▼
Diabetes Prediction
```

---

## 📊 Input Features

The application uses the following patient information:

| Feature             | Description               |
| ------------------- | ------------------------- |
| Gender              | Patient gender            |
| Age                 | Patient age               |
| Hypertension        | Presence of hypertension  |
| Heart Disease       | Presence of heart disease |
| Smoking History     | Patient smoking history   |
| BMI                 | Body Mass Index           |
| HbA1c Level         | HbA1c blood level         |
| Blood Glucose Level | Blood glucose level       |

---

## ⚙️ Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
ANN Model Creation
   ↓
Sigmoid Activation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit UI
   ↓
Real-Time Prediction
```

---

## 🔬 Model Training

The ANN model uses:

```python
Dense(16, activation="sigmoid")
Dense(8, activation="sigmoid")
Dense(1, activation="sigmoid")
```

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 📈 Prediction

The model returns a probability between **0 and 1**.

```text
Probability >= 0.5
        ↓
Diabetes

Probability < 0.5
        ↓
No Diabetes
```

The Streamlit application displays the prediction probability to the user.

---

## 🖥️ Streamlit Application

The Streamlit interface allows users to enter:

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

After clicking **Predict Diabetes**, the application processes the input and generates the ANN prediction.

---

## 📁 Project Structure

```text
Diabetes-ANN/
│
├── app.py
├── diabetes_ann_model.keras
├── scaler.pkl
├── columns.pkl
├── diabetes_dataset.csv
├── requirements.txt
└── README.md
```

---

## 💾 Model Files

### `diabetes_ann_model.keras`

Contains the trained ANN model.

### `scaler.pkl`

Contains the trained `StandardScaler` used for feature scaling.

### `columns.pkl`

Contains the training feature column names to ensure the Streamlit input matches the ANN training data.

---

## ▶️ Run Project Locally

### Step 1: Clone Repository

```bash
git clone [ADD YOUR GITHUB REPOSITORY LINK HERE]
```

### Step 2: Open Project Folder

```bash
cd Diabetes-ANN
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

```text
streamlit
tensorflow
pandas
numpy
scikit-learn
joblib
```

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how an **Artificial Neural Network with Sigmoid activation** can be used for a binary classification problem and integrated into an interactive web application using Streamlit.

---

## 🔮 Future Improvements

* Add more advanced ANN architectures
* Hyperparameter tuning
* Add model performance dashboard
* Add ROC-AUC curve
* Add prediction history
* Improve UI/UX
* Deploy with Streamlit Cloud
* Add additional health-related visualizations

---

## ⚠️ Disclaimer

This application is created for **educational and demonstration purposes only**.

The predictions generated by this application should not be considered a medical diagnosis or a substitute for professional medical advice.

---

## 👨‍💻 Author

**Ritesh Kumar Kasaudhan**

MCA Student | Aspiring Data Analyst | Machine Learning Enthusiast

---

## 🔗 Links

🌐 **Live Demo:**(https://ritesh-diabetes.streamlit.app/)

💻 **GitHub Repository:**(https://github.com/ritesh975/Diabetes-Prediction-using-ANN)

📊 **Project:** Diabetes Prediction using ANN

---

## ⭐ If you found this project useful

Consider giving this repository a ⭐ on GitHub!
