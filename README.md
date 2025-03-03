# 🍷 Machine Learning Project - Clustering Wine Chemical Profiles

## 📌 Introduction

This project aims to analyze **the chemical composition of wines** and identify **natural groupings** using **unsupervised clustering techniques**.  
Through clustering, we explore whether wines can be categorized **without predefined labels**, providing insights into **similarities and differences** in their composition.  

## 🎯 Objectives

✅ Identify **natural clusters** of wines based on their chemical properties.  
✅ Compare multiple **clustering algorithms** to select the best-performing model.  
✅ Evaluate if **traditional wine categories** emerge naturally from the clustering process.  
✅ Develop and deploy a **Flask-based web application** that allows users to classify wines using the trained model.

---

## 📊 Dataset

- **Source**: The dataset consists of **chemical properties** of wines.
- **Features**: The dataset includes **13 chemical components**, such as:
  - Alcohol
  - Malic Acid
  - Ash
  - Magnesium
  - Flavonoids
  - Proline  
  - ... *(see full list in `app.py`)*

---

## 📖 Methodology

### 🔍 1. Data Exploration & Preprocessing

- Performed **data cleaning** and handled missing values.
- Conducted **descriptive statistics** and visualizations.
- Computed **correlation matrices** to understand feature relationships.
- Applied **Principal Component Analysis (PCA)** to reduce dimensionality.

### 🏆 2. Clustering Algorithms Tested

| Algorithm | Advantages | Disadvantages |
|-----------|------------|--------------|
| **K-Means** | Fast, interpretable, efficient for well-separated clusters | Sensitive to initialization, assumes spherical clusters |
| **DBSCAN** | Detects clusters of various shapes | Highly sensitive to hyperparameters (epsilon, MinPts) |
| **Hierarchical (CAH)** | No need to predefine `k` clusters | Computationally expensive for large datasets |
| **GMM (Gaussian Mixture Model)** | Handles probabilistic distributions | Less effective for dense clusters |
| **BIRCH** | Efficient on large datasets | Limited to ellipsoidal cluster shapes |

📌 **Final Model Selected: K-Means**  
✅ Forms **well-defined, compact clusters**  
✅ Computationally **efficient**  
✅ **Easy to interpret** for analysis and predictions  

### ⚙️ 3. Model Evaluation

The models were evaluated using key clustering metrics:

- **Silhouette Score**: Measures **cluster separation** (higher = better).
- **Calinski-Harabasz Score**: Evaluates **cluster compactness** (higher = better).
- **Davies-Bouldin Score**: Measures **cluster dispersion** (lower = better).

📈 **K-Means performed best**, creating three distinct wine clusters.  
📌 **DBSCAN failed**, classifying **100% of points as noise**.

---

## 🚀 Web Application

The project includes a **Flask-based web app** where users can input **chemical values** of a wine and get **its predicted cluster**.

### 🖥️ Features:

✅ **User-friendly form** to input chemical characteristics.  
✅ **Real-time prediction** of wine cluster using **trained K-Means model**.  
✅ **Automated preprocessing** (scaling, PCA transformation).  
✅ **Flask-based deployment** for easy usage.

---

## 📂 Project Structure

/ML_Wine_Clustering  
│── app.py # Flask app for predictions  
│── kmeans_model.pkl # Pre-trained K-Means model  
│── scaler.pkl # Standard scaler for normalization  
│── pca.pkl # Pre-trained PCA model for dimensionality reduction  
│── templates/  
│ ├── form.html # User input form  
│ ├── result.html # Displays predicted cluster  
│── static/ # CSS, JavaScript, images  
│── Projet_ML.ipynb # Jupyter Notebook with full implementation  
│── README.md # Project documentation  

---
## 🚀 Deployment Instructions

### 1️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
2️⃣ Run Flask Application
bash
 
 
python app.py
3️⃣ Access Web Interface
Open http://127.0.0.1:5000/ in a browser.

🔧 Future Improvements
🚀 Preprocessing Enhancements:

Noise reduction and non-linear transformations (Box-Cox, Power Transform).
Hyperparameter tuning using grid search and cross-validation.
⚡ Alternative Algorithms:

OPTICS, Graph-Based Clustering, Spectral Clustering.
Hybrid approach combining ICA + PCA.
📌 Scalability:

Implementing distributed clustering (e.g., MiniBatch K-Means).
Deploying the model as a cloud-based API.
👤 Author
Marwane KASSA


📜 License
This project is open-source and free to use.

🔍 Developed as part of a machine learning research on wine clustering.

yaml

---
