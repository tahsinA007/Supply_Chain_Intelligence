# Supply_Chain_Intelligence
An end-to-end machine learning project for predicting inventory stockout risk and supporting supply chain decision-making. Hence it is predicting Estimated Delivery Time, Delivery status(early or late etc), Shipment Profile , grouping data points based on their density and identifies points in low-density regions as noise or potential anomalies

**To get full markdown & codes of kaggle notebook of this project, check out .ipynb file attached in the file section of this repo.**

**End to end data pipeline - - -**
Data Collection
       ->
Exploratory Data Analysis (EDA)
       ->
Data Preprocessing for regression
       ->
Regression models and metrics
       ->
Final Model Selection - Regression(SVR)
       ->
Data Preprocessing for Classification
       ->
Classification models and metrics
       ->
Final Model Selection - Classification(XG Boost Classifier)
       ->
K Means for shipment profile
       ->
DBSCAN for anomaly checking
       ->
Checking new input data prediction

**Deployment flow - - -**
Develop ML Pipeline
        ->
Save Trained Model & Preprocessor
        ->
Create Streamlit App
        ->
Add requirements.txt
        ->
Push Project to GitHub
        ->
Deploy on Streamlit Community Cloud
        ->
Generate Public App URL
        ->
Live Streamlit Application

## 🛠️ Tools & Technologies
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Techniques:** Anomaly Detection, K-Means Clustering, Regression, Classification, Feature Engineering
- **Model Persistence:** Joblib
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Development:** Jupyter Notebook
- **Version Control:** Git, GitHub
  
🚀 **[Live Demo](https://supply-chain-risk-prediction.streamlit.app/)**
