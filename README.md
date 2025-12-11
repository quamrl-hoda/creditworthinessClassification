
# 💳 Credit Score Prediction System  
### Based on Loan Approval Dataset (Machine Learning Project)

This project predicts the **Credit Score (Good / Poor)** of a customer using their loan application details.  
Since the dataset does not contain a credit score column, we derive it from **Loan_Status**:

- **Loan_Status = Y → Credit Score: Good**
- **Loan_Status = N → Credit Score: Poor**

This system uses a **machine learning classification model**, trained using Logistic Regression, and provides a **Streamlit UI** for real-time predictions.

---

## 📌 Project Structure

```

project/
│── app.py              # Streamlit UI
│── requirements.txt
│── README.md
│── dataset/
│     ├── loandataset.csv
│── model/
│     ├── logistic_pipeline.joblib
│     ├── preprocessor.joblib
│     ├── rf_pipeline.joblib

````

---

##  Dataset Information

The dataset contains the following important features:

- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  
- ApplicantIncome  
- CoapplicantIncome  
- LoanAmount  
- Loan_Amount_Term  
- Credit_History  
- Property_Area  
- Loan_Status (used to generate Credit Score label)

---

## Data Preprocessing Steps

1. Missing values handled using median strategy  
2. Label encoding for categorical features  
3. Standard scaling for numeric features  
4. Creation of **Credit_Score** from **Loan_Status**  
5. Splitting dataset into train/test sets  

---

## Machine Learning Model

We use **Logistic Regression**, which is ideal for binary classification problems.

### Model Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve  

All evaluation plots are saved in `/output/`.

---

##  Running the Project

### **1. Install required libraries**
```bash
pip install -r requirements.txt
````

### **2. Train the ML Model**

```bash
python model_training.py
```

This will create:

* `logistic_model.joblib`
* `scaler.joblib`
* `encoders.joblib`

### **3. Run the Streamlit UI**

```bash
streamlit run app.py
```

Open in browser:
 [http://localhost:8501](http://localhost:8501)

---

## Streamlit Application
The app takes user input:

* Gender
* Married status
* Income values
* Loan amount
* Education level
* Property area
* Credit history

Predicts:

**"Good"** or **"Poor"** credit score.

---

## Requirements

```
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
```

---

## Results

* The model predicts creditworthiness based on financial and demographic factors.
* Helps banks and financial institutions assess customer risk.
* Easy-to-use UI allows real-time prediction.

---

## Conclusion

This project demonstrates the complete workflow of:

✔ Data preprocessing
✔ Feature encoding & scaling
✔ ML model training
✔ Model saving
✔ Live prediction UI with Streamlit

It is suitable for:

* Academic projects
* Machine learning coursework
* Banking/finance risk assessment demonstrations
* Portfolio projects

---

##  Author

**Quamrul Hoda**
B.Tech – AIML

