# ✈️ Learning-Based Airline Delay Estimation

## 🧠 Overview
Flight delays are a persistent challenge in the aviation industry, impacting airline efficiency and passenger satisfaction.  
This project leverages **Machine Learning (ML)** and **Deep Learning (DL)** techniques using **Apache Spark** to predict flight delays based on large-scale flight data.

The objective is to build robust predictive models that can help airlines **anticipate delays**, optimize scheduling, and enhance the passenger experience.

---

## 👥 Team
**Team Name:** GOAT  
**Members:**  
- Yuvaneswaren Ramakrishnan Sureshbabu *(Team Leader)*  
- Santhosh Reddy Katasani Venkata  
- Arunaswin Gopinath  
- Pulipati Kushank  

---

## 📊 Dataset
**Source:** U.S. Department of Transportation — *Airline Delay Prediction Dataset (2014)*  
**Records:** 5,819,811  
**Attributes:** 18 features (numerical, categorical, temporal)  
**File Size:** ~300 MB (CSV)

Key attributes include:
- Flight times (DepTime, ArrTime)
- Delays (DepDelay, ArrDelay)
- WeatherDelay, CarrierDelay, NASDelay
- Distance, Carrier code, Origin, Destination
- Date, Month, Day of Week

---

## 🔍 Exploratory Data Analysis (EDA)
Exploratory analysis was performed to uncover:
- Delay trends across airports and routes
- Monthly and hourly delay distributions
- Contributions of delay causes (weather, carrier, etc.)
- Seasonal and regional delay variations

Key visualizations:
- U.S. airport delay heatmap  
- Flight network visualization  
- Top 15 delayed routes  
- Delay distribution by day/hour  

---

## ⚙️ Data Preprocessing
1. **Handling Missing Data**
   - Replaced missing delay values (e.g., `CarrierDelay`, `WeatherDelay`) with `0`
   - Removed rows missing critical info like `DepTime` or `ArrTime`

2. **Addressing Class Imbalance**
   - Applied **oversampling** to minority (delayed) class  
   - Used **class weight adjustments** in models (e.g., Logistic Regression)

3. **Feature Engineering**
   - Created a binary *Delayed* target variable  
   - Added temporal features (Month, Day of Week)  
   - Categorized flight distances  
   - Encoded categorical features (e.g., airlines, airports)

---

## 🧩 Machine Learning Models
### Models Implemented
- **Logistic Regression**
  - Baseline model with balanced dataset  
  - Accuracy: 64.86%, F1 Score: 0.6476  

- **Random Forest Classifier**
  - Highest recall (0.7444), AUC-ROC: 0.6942  
  - Best at detecting delayed flights  

- **Decision Tree Classifier**
  - Accuracy: 67.8% (highest), but lowest AUC-ROC (0.5386)  

**✅ Observation:**  
Among ML models, **Random Forest** performed best for delay detection.

---

## 🤖 Deep Learning Models
To improve predictive accuracy, three deep learning models were developed:

### 1. **DNN (Fully Connected Network)**
- Architecture: 4 hidden layers (64 → 128 → 64 → 32)
- Activation: ReLU (hidden), Sigmoid (output)
- Accuracy: **99.16%**

### 2. **CNN (Convolutional Neural Network)**
- Layers: Conv1D + MaxPooling + Dense
- Optimizer: Adam, Loss: Binary Crossentropy  
- Accuracy: **98.34%**

### 3. **CRNN (Convolutional Recurrent Neural Network)**
- Combined **Conv1D** for spatial features + **LSTM** for temporal dependencies  
- Accuracy: **98.41%**

**🏆 Best Performing Model:**  
The **DNN** outperformed all others, capturing complex relationships and achieving near-perfect accuracy.

---

## 💡 Results Summary

| Model Type | Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------------|--------|-----------|------------|---------|-----------|----------|
| ML | Logistic Regression | 64.9% | 0.6496 | 0.6457 | 0.6476 | 0.7041 |
| ML | Random Forest | 64.4% | 0.6131 | **0.7444** | 0.6724 | 0.6942 |
| ML | Decision Tree | **67.8%** | 0.6049 | 0.6867 | 0.6432 | 0.5386 |
| DL | DNN | **99.16%** | - | - | - | - |
| DL | CNN | 98.34% | - | - | - | - |
| DL | CRNN | 98.41% | - | - | - | - |

---

## 🧭 Key Insights
- Deep learning models (especially DNN) significantly outperform classical ML models.  
- Class imbalance correction and feature engineering improved reliability.  
- Random Forest remains strong for quick, interpretable predictions.  
- Deep models can generalize better for operational deployment.

---

## 🚀 Future Enhancements
- Implement **Transformer-based architectures** for improved temporal modeling.  
- Incorporate **real-time weather & traffic data**.  
- Explore **batch normalization** and **dropout tuning** to further reduce overfitting.  
- Deploy as an **API service** for live airline delay predictions.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frameworks:** PySpark, TensorFlow, Keras, Scikit-learn  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, VS Code  
