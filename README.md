# 🧠 Extracting Skills from Job Postings Using Data Mining

> Uncover in-demand tech skills by mining LinkedIn job postings using NLP, TF-IDF, K-Means Clustering, and Association Rule Mining (FP-Growth).

---

## 📘 Overview

This project aims to identify the most relevant and frequently co-occurring skills in **tech-related job postings** by mining job descriptions using a multi-stage data mining pipeline.

It combines:
- **Text Preprocessing + TF-IDF** for skill weighting
- **K-Means Clustering** for job segmentation
- **Association Rule Mining (FP-Growth)** for uncovering common tech stacks

📁 Dataset: [LinkedIn Job Postings - Kaggle](https://www.kaggle.com/datasets)  
🔍 Focus: Computer Science / Tech roles  
🧪 Libraries: Scikit-learn, MLxtend, Pandas, Matplotlib

---

## 🧱 Methods Used

### 1. 🧹 Preprocessing Pipeline
- Custom tokenization and stopword filtering
- Focused only on **CS/Tech** job segments
- Final format: tokenized skill lists by job ID

### 2. 🧮 TF-IDF Feature Extraction
- Converts text into a Job × Skill weight matrix
- Highlights skill uniqueness within postings

### 3. 📊 K-Means Clustering
- **K=13** (chosen via Elbow Method)
- Segmented job market into interpretable profiles:
  - Backend, Frontend, Data Science, DevOps, etc.

### 4. 🔍 Association Rule Mining
- Used **FP-Growth** for frequent pattern extraction
- Mined skill sets that commonly appear together in job postings

---

## 📊 Sample Results

### 📌 Clustered Job Profiles

| Cluster | Profile        | Top Skills                          |
|---------|----------------|-------------------------------------|
| 3       | Backend         | Java, Spring, API, Microservices    |
| 6       | Data Science    | Python, ML, Pytorch, TensorFlow     |
| 11      | Frontend        | JavaScript, React, HTML, CSS        |
| 13      | Cloud/DevOps    | AWS, Docker, Kubernetes, CI/CD      |

### 🔗 Top Association Rules

| IF Skills         | THEN Skills | Confidence | Lift |
|------------------|-------------|------------|------|
| {typescript}     | {react}     | 0.85       | 1.83 |
| {aws, serverless}| {lambda}    | 0.92       | 2.15 |
| {spark, hadoop}  | {scala}     | 0.77       | 1.51 |

---

## 📚 Key Learnings

- Applying **text mining** on unstructured job postings  
- Using **TF-IDF** and **K-Means** for unsupervised insights  
- Extracting association rules via **FP-Growth**  
- Translating numeric analysis (TF-IDF, Lift) into **market trends**

---

## ▶️ Run It Yourself

```bash
# Install dependencies
pip install -r requirements.txt

# Run main scripts
python data_preprocessing_pipeline.py
python kmeans_clustering.py
python skill_association_mining.py
