# 🎓 Student Dropout Prediction & Educational Early-Warning System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9266-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-88.59%25-brightgreen.svg)]()

An end-to-end Machine Learning system and interactive Early-Warning Web Application built with **Logistic Regression** to identify students at risk of dropping out. By analyzing academic performance, financial standing, demographic factors, and enrollment attributes, the system empowers academic institutions to deploy proactive retention interventions before dropout occurs.

---

## 📌 Project Overview & Early-Warning Objective

High dropout rates in higher education carry severe financial, institutional, and human costs. Traditional retention efforts are often reactive—identifying students only after they have failed or withdrawn. 

This project implements an **Educational Early-Warning System (EWS)** using **Logistic Regression** to predict a continuous dropout risk probability $P(\text{Dropout} | X)$ for each student based on:
1. **Academic Performance**: 1st & 2nd semester approved curricular units, evaluation grades, and admission performance.
2. **Financial Standing**: Tuition fee payment status (up-to-date vs in arrears), scholarship holder status, and debtor status.
3. **Demographic & Socioeconomic Context**: Age at enrollment, gender, displacement status, application mode, and parental qualification levels.

---

## 📐 Mathematical Formulation of Logistic Regression

Logistic Regression models the probability $p = P(Y=1 | X)$ of a binary event (Student Dropout = 1) using the **Sigmoid / Logistic Function**:

$$P(Y = 1 | X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of input features $X_1, X_2, \dots, X_n$ and weights $\beta_0, \beta_1, \dots, \beta_n$:

$$z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n$$

### Log-Odds (Logit Link Function)
The model represents the natural logarithm of the odds of dropout:

$$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \sum_{i=1}^n \beta_i X_i$$

- A **positive coefficient ($\beta_i > 0$)** increases the log-odds (and probability) of student dropout (e.g. tuition fees in arrears, higher enrollment age).
- A **negative coefficient ($\beta_i < 0$)** decreases the log-odds of dropout (e.g. higher number of approved units in 2nd semester, holding a scholarship).

### Log-Loss Cost Function
The model parameters $\mathbf{\beta}$ are optimized using Maximum Likelihood Estimation via the **Binary Cross-Entropy Loss**:

$$J(\mathbf{\beta}) = -\frac{1}{N} \sum_{j=1}^N \left[ y_j \ln(p_j) + (1 - y_j) \ln(1 - p_j) \right]$$

---

## 🚀 Key Project Phases & Screenshot Proofs

### Phase 2: Dataset Collection & Exploration
- **Dataset**: UCI Predict Students Dropout and Academic Success Dataset (4,424 records, 36 features).
- **Target Distribution**: Graduate (49.9%), Dropout (32.1%), Enrolled (18.0%).
- **Screenshot Proof**: `visuals/phase2_dataset_exploration.png`

### Phase 3: Data Cleaning & Preprocessing
- **Missing Value Analysis**: Verified 0 missing values across all features.
- **Target Binarization**: Mapped `Dropout` $\rightarrow 1$ (At Risk) and `Graduate`/`Enrolled` $\rightarrow 0$ (Retained).
- **Standardization**: Feature matrix transformed using `StandardScaler` ($\mu=0, \sigma=1$).
- **Train/Test Split**: 80% Training (3,539 samples), 20% Test (885 samples), stratified.
- **Screenshot Proof**: `visuals/phase3_data_preprocessing.png`

### Phase 4: Exploratory Data Analysis (EDA)
Key findings:
- **Tuition Fees Status**: Students in arrears have an **86.6% dropout rate** vs **24.7%** for students up to date.
- **Scholarship Status**: Scholarship holders demonstrate a low **12.2% dropout rate** vs **38.7%** for non-holders.
- **Academic Units Approved**: Retained students average **5.62 approved units** in 2nd semester vs **1.94** for dropouts.
- **Screenshot Proof**: `visuals/phase4_eda.png`

### Phase 5: Logistic Regression Model Training
- **Model**: `LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)`
- **Top Risk Drivers (+ Coefficients)**:
  - `Curricular units 2nd sem (enrolled)` (+0.8106)
  - `Age at enrollment` (+0.3580)
- **Top Protective Factors (- Coefficients)**:
  - `Curricular units 2nd sem (approved)` (-1.8252)
  - `Curricular units 1st sem (approved)` (-0.9264)
  - `Tuition fees up to date` (-0.7426)
- **Screenshot Proof**: `visuals/phase5_model_training.png`

### Phase 6: Model Evaluation & Performance Metrics
- **Accuracy**: **88.59%**
- **Precision**: **88.94%**
- **Recall**: **73.59%**
- **F1-Score**: **80.54%**
- **ROC-AUC**: **0.9266 (92.66%)**

| Metric | Score | Educational EWS Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | 88.59% | Overall correct retention vs dropout predictions |
| **Precision** | 88.94% | High reliability when flagging a student as "At Risk" |
| **Recall** | 73.59% | Proportion of actual dropouts caught by the system |
| **ROC-AUC** | 92.66% | Excellent probability separation capability |

#### False Positive vs False Negative Educational Impact
- **False Positives (26 cases / 2.9%)**: Student predicted at risk who completes studies. Low operational cost (extra advising call).
- **False Negatives (75 cases / 8.5%)**: Student predicted safe who drops out. High cost in EWS (lost student opportunity).
- **Screenshot Proof**: `visuals/phase6_model_evaluation.png`

---

## 💻 Phase 7: Student Risk Prediction Web Application

An interactive web application built with **Streamlit** allows academic advisors and counselors to enter student details and immediately receive risk predictions.

### Key Application Features:
1. **Live Probability Meter & Dynamic Risk Badges**:
   - 🟢 **Low Risk**: $<30\%$ Dropout Probability
   - 🟡 **Moderate Risk**: $30\% - 65\%$ Dropout Probability
   - 🔴 **High Risk**: $>65\%$ Dropout Probability
2. **Actionable Counseling Advice**: Tailored intervention strategies based on risk severity (academic tutoring, financial grant aid, attendance tracking).
3. **Pre-configured Test Case Scenarios**: Quick load sidebar buttons for:
   - **Scenario A**: High Risk Student Profile
   - **Scenario B**: Moderate / Borderline Student Profile
   - **Scenario C**: Low Risk / High Performing Student Profile
- **Screenshot Proofs**: `visuals/phase7_app_high_risk.png`, `visuals/phase7_app_moderate_risk.png`, `visuals/phase7_app_low_risk.png`

---

## 🛠️ Project Architecture & File Structure

```
student-dropout-prediction/
│
├── data/
│   ├── student_dropout_raw.csv         # Downloaded UCI raw dataset
│   └── processed_student_data.csv       # Preprocessed feature matrix
│
├── models/
│   ├── logistic_regression_model.pkl    # Serialized Logistic Regression model
│   ├── scaler.pkl                       # StandardScaler object
│   └── feature_names.pkl                # Feature order list
│
├── visuals/                             # Screenshot Proofs & Plots
│   ├── phase2_dataset_exploration.png
│   ├── phase3_data_preprocessing.png
│   ├── phase4_eda.png
│   ├── phase5_model_training.png
│   ├── phase6_model_evaluation.png
│   ├── phase7_app_high_risk.png
│   ├── phase7_app_moderate_risk.png
│   └── phase7_app_low_risk.png
│
├── src/
│   ├── dataset.py                       # Dataset loader & synthetic benchmark generator
│   ├── preprocessing.py                 # Missing value handling & scaling
│   ├── model.py                         # Logistic Regression model training
│   └── evaluation.py                    # Classification metrics & confusion matrix
│
├── 01_dataset_exploration.py            # Phase 2 runner
├── 02_data_preprocessing.py            # Phase 3 runner
├── 03_eda.py                            # Phase 4 runner
├── 04_model_training.py                 # Phase 5 runner
├── 05_model_evaluation.py               # Phase 6 runner
├── app.py                               # Phase 7 Streamlit Web Application
├── README.md                            # Complete documentation
└── requirements.txt                     # Project dependencies
```

---

## 📥 Installation & How to Run

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts (Phases 2-6)
Run all phase execution scripts sequentially:
```bash
python 01_dataset_exploration.py
python 02_data_preprocessing.py
python 03_eda.py
python 04_model_training.py
python 05_model_evaluation.py
```

### 4. Launch Interactive Web Application (Phase 7)
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Deployment Overview (Phase 8)

The application is structured for cloud deployment:
1. **Streamlit Community Cloud**:
   - Connect GitHub repository.
   - Set main file path to `app.py`.
   - Deploy with 1 click.
2. **Docker Container Deployment**:
   - `Dockerfile` provided for deploying on AWS App Runner, Heroku, or GCP Cloud Run.

---

## 📜 License & Acknowledgments
- Dataset source: [UCI Machine Learning Repository - Predict Students Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (Dataset ID 697).
- Built with Python, Scikit-Learn, Streamlit, Matplotlib, and Seaborn.
# Student-dropout-prediction-main
