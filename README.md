# 🎓 Student Performance Prediction: An End-to-End Data analysis Project 🚀

![GitHub stars](https://img.shields.io/github/stars/farhad-here/Predict_student_performance?style=social)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-lightgrey?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Project Summary

This project demonstrates a complete and practical approach to data science, with the goal of **predicting student academic performance** based on various social, family, and educational factors. Through this project, a full data pipeline has been created, covering everything from exploratory data analysis (EDA) to the construction and evaluation of a machine learning model.

This project can help educators and institutions proactively identify students at risk of underperformance, allowing for timely and effective interventions.

---

## 📓 Description
An end-to-end data analysis and machine learning project for predicting student performance using educational and social data. The project includes data analysis, model building, and an interactive dashboard.
---

## 🎯 The Problem

Student academic success is influenced by numerous factors. By analyzing data related to study habits, family status, and other behavioral characteristics, we can build a model that predicts which students might receive a low final grade. This prediction enables a proactive, preventative approach rather than a reactive one.

---

## 🛠️ Project Architecture & Workflow

This project follows a structured and modular workflow:

1.  **Data Collection & Cleaning:**
    * The dataset is loaded from a CSV file, and cleaning and preprocessing operations (such as handling missing values and correcting data types) are performed.

2.  **Exploratory Data Analysis (EDA):**
    * A comprehensive statistical and visual analysis is conducted to uncover hidden relationships between features and the final student grade.
    * Visualizations such as a Correlation Matrix, Histograms, and Scatter Plots are created for a better understanding of the data.

3.  **Feature Engineering & Preprocessing:**
    * Categorical features are encoded numerically to make them suitable for machine learning models.
    * The data is split into training and testing sets.

4.  **Machine Learning Model:**
    * A classification model is trained to predict the final student grade (e.g., pass or fail).
    * The model's performance is measured using appropriate evaluation metrics (such as accuracy, F1-Score, and a Confusion Matrix).

5.  **Conclusion & Insights:**
    * A summary of key findings and important insights derived from the data analysis is presented.

---

## 💻 Technical Stack & Libraries

* **Programming Language:** Python
* **Data Analysis:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-learn`
* **Development Environment:** `Jupyter Notebook`

---

## 🏙️ Dataset
The dataset includes features like:
- **Demographic**: `sex`, `age`, `address`, `famsize`
- **Family and Education**: `medu`, `fedu`, `mjob`, `fjob`
- **Lifestyle and Social**: `famrel`, `freetime`, `goout`, `dalc`, `walc`, `health`
- **Academic**: `absences`, `g1`, `g2`, and the target `g3`


---
## ✋ Approach

- **Regression**  
  - Predict `g3` as a numerical value (range: 0–20).
- **Classification**  
  - Convert `g3` into two classes:  
    - **Fail**: 0–9  
    - **Pass**: 10–20  

---

## ▶️ How to Run the Project

Follow these steps to run the project on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/farhad-here/Predict_student_performance.git](https://github.com/farhad-here/Predict_student_performance.git)
    cd Predict_student_performance
    ```

2.  **Install Dependencies:**
    * It is highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Project:**
    * Open the `Jupyter Notebook` file and run all the cells in order.
    ```bash
    jupyter notebook
    ```


## ✉️ PowerBi Dashboard

<img width="2071" height="1165" alt="powerstudenbi" src="https://github.com/user-attachments/assets/b7e9b132-a3b5-458b-86e6-c925fb3f9965" />

