# 🛒 Social Network Ads Purchase Prediction using SVM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Demo
Try out the interactive web application here: 
**[Launch the Gradio App on Hugging Face Spaces](https://huggingface.co/spaces/MalikShehram/Social-Ads-Purchase-Predictor)**

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing & Feature Engineering](#data-preprocessing)
5. [Model Architecture: Support Vector Machine](#model-architecture)
6. [Model Accuracy & Performance](#model-accuracy)
7. [Visualizing the Decision Boundary](#visualizations)
8. [Installation & Usage](#installation)
9. [Future Improvements](#future-improvements)
10. [Conclusion](#conclusion)

---

## 🚀 Project Overview <a name="project-overview"></a>
This project leverages Machine Learning to predict customer purchase behavior based on targeted social media advertisements. By analyzing historical ad-click and conversion data, we built a robust predictive model using a **Support Vector Machine (SVM)** classifier. 

The primary goal of this repository is to demonstrate an end-to-end classification pipeline—from data ingestion and feature scaling to model training, accuracy evaluation, and complex decision-boundary visualization.

---

## 💼 Business Problem <a name="business-problem"></a>
Marketing budgets are finite. When a company launches a new product (e.g., a luxury car, a tech gadget, or a premium service), it cannot afford to show advertisements to every user on a social network. 

**The Challenge:** How can a business optimize its advertising spend to target only the users who are most likely to click the ad and make a purchase?
**The Solution:** Build a predictive model that identifies the demographic traits (like Age and Income) of users who convert, allowing the marketing team to hyper-target similar profiles in future campaigns.

---

## 📊 Dataset Description <a name="dataset-description"></a>
The project utilizes the `Social_Network_Ads.csv` dataset, which simulates a company's social media user base and their response to a specific product advertisement.

The dataset consists of 400 user records with the following columns:
* **User ID:** A unique identifier for the social network user (Dropped during training as it lacks predictive power).
* **Gender:** The gender of the user (Male/Female).
* **Age:** The age of the user in years.
* **EstimatedSalary:** The estimated annual income of the user.
* **Purchased:** The target variable / label (`0` = Did not purchase, `1` = Purchased).

---

## ⚙️ Data Preprocessing & Feature Engineering <a name="data-preprocessing"></a>
To ensure our SVM model achieves maximum accuracy, strict preprocessing steps were applied:

### 1. Feature Selection
While Gender can be a factor, EDA usually reveals that **Age** and **Estimated Salary** are the most dominant predictors for this specific dataset. The features were carefully selected to isolate the strongest signals.

### 2. Label Encoding
Categorical variables (like Gender) were transformed into numerical formats using `LabelEncoder`. This ensures the mathematical equations within the ML algorithms can process the data without throwing string-to-float conversion errors.

### 3. Train/Test Split
The data was split using an 75-25 ratio:
* **Training Set:** 300 records used to train the SVM model.
* **Test Set:** 100 unseen records used to objectively evaluate the model's accuracy.

### 4. Feature Scaling (Crucial for SVM)
Support Vector Machines are highly sensitive to the scale of the input features because they rely on calculating distances (like Euclidean distance) between data points. 
* A salary of $76,000 would statistically dominate an age of 27 without scaling.
* We applied `StandardScaler` to standardize the features so they have a mean of 0 and a standard deviation of 1.

---

## 🧠 Model Architecture: Support Vector Machine <a name="model-architecture"></a>
We utilized the `SVC` (Support Vector Classifier) module from the `scikit-learn` library. 

**Why SVM?**
SVM is a powerful algorithm that seeks to find the "hyperplane" that best separates the two classes (Purchasers vs. Non-Purchasers) with the maximum margin. 

**The RBF Kernel:**
In the real world, human behavior is rarely linearly separable. A simple straight line cannot accurately divide younger, high-income buyers from older, low-income buyers. Therefore, we used the **Radial Basis Function (RBF) Kernel**. 
The RBF kernel projects our 2D data into a higher-dimensional space, allowing the model to draw highly complex, non-linear boundaries around our target clusters. This is the secret behind the model's high accuracy.

---

## 🎯 Model Accuracy & Performance <a name="model-accuracy"></a>
Accuracy is the most critical metric for a predictive business model. Due to the combination of effective Feature Scaling and the non-linear RBF kernel, this model achieves highly impressive results.

### Expected Accuracy
When training the SVM (RBF kernel) on the scaled Age and Estimated Salary features, the model typically achieves an **Accuracy of ~90% to 93%** on the unseen test set.

### Evaluation Metrics Used
* **Model Score (`classifier.score`)**: Returns the mean accuracy on the given test data and labels. 
* **Confusion Matrix**: We utilized `confusion_matrix` from `sklearn.metrics` to break down the exact performance:
  * **True Positives (TP):** Users the model correctly predicted would buy.
  * **True Negatives (TN):** Users the model correctly predicted would NOT buy.
  * **False Positives (FP - Type I Error):** Users predicted to buy, but didn't.
  * **False Negatives (FN - Type II Error):** Users predicted NOT to buy, but actually did.

By minimizing False Positives, the business ensures it doesn't waste ad spend. By minimizing False Negatives, the business ensures it doesn't miss out on potential revenue. The RBF SVM provides an excellent balance of both.

---

## 📈 Visualizing the Decision Boundary <a name="visualizations"></a>
One of the standout features of this repository is the inclusion of advanced `matplotlib` visualizations using `ListedColormap`. 

The code generates high-resolution contour plots for both the Training and Test sets:
* **Red Regions:** The area where the model predicts a user will NOT purchase (Class 0).
* **Green Regions:** The area where the model predicts a user WILL purchase (Class 1).
* **Data Points:** Actual users plotted by their Age and Salary.

Because we used an RBF kernel, you will notice the decision boundary is smooth and circular/curved, perfectly capturing the "pockets" of specific demographics that are highly likely to convert!

---

## 💻 Installation & Usage <a name="installation"></a>

### Prerequisites
Ensure you have Python installed. You will also need Jupyter Notebook or JupyterLab to run the `.ipynb` file.

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MalikShehram512/social-ads-svm-prediction.git](https://github.com/MalikShehram512/social-ads-svm-prediction.git)
   cd social-ads-svm-prediction
