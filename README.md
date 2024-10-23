# Student Dropout Prediction Web Application

This project aims to predict student dropout using machine learning models integrated into a web application developed with Streamlit. A complementary dashboard application has also been created to provide insights into the dataset and help analyze trends related to student dropouts. The project is organized into weekly segments, each focusing on different stages of development, from data preprocessing to final deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Project Workflow](#project-workflow)
  - [Data Preprocessing](#data-preprocessing-week-1)
  - [Dashboard Creation](#dashboard-creation-week-2)
  - [Feature Engineering](#feature-engineering-week-3)
  - [Model Training and Evaluation](#model-training-and-evaluation-week-4)
  - [Hyperparameter Tuning](#hyperparameter-tuning-week-5)
  - [Deployment](#deployment-week-6)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The **Student Dropout Prediction** project is designed to identify students at risk of dropping out based on academic, demographic, and socio-economic data. The project is developed as a machine learning pipeline that includes a web application for predictions and a dashboard for visualizing key data trends.

## Folder Structure
The project is divided into weekly folders, each focusing on different aspects of the development process:

- **Week 1: Data Preprocessing**  
  - Data cleaning, handling missing values, scaling, and transformation.
- **Week 2: Dashboard Creation**  
  - Creation of the dashboard for visualizing dataset trends using Plotly and Streamlit. Other statistical analysis is also carried out such as hypothesis testing.
- **Week 3: Feature Engineering**  
  - Engineering new features based on academic and socio-economic factors. Interaction and polynomial features are also created with the aid of a custom scikit-learn transformer.
- **Week 4: Data Modelling**  
  - Model training and evaluation using various machine learning and deep learning algorithms.
- **Week 5: Hyperparameter Tuning**  
  - Optimization of model performance using techniques like GridSearchCV, RandomSearchCV and BayesSearchCV.
- **Week 6: Deployment**  
  - Deployment of the web application and dashboard using Streamlit.

## Features
- **Predict Dropout Risk**: Input student details to predict the likelihood of dropout.
- **Visualize Data**: Explore the dataset via an interactive dashboard that displays key features, trends, and insights.
- **Feature Importance**: Understand which factors contribute most to predicting student dropout.
- **End-to-End Pipeline**: From data preprocessing to deployment, everything is included.

## Tech Stack
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning Libraries**: Scikit-learn, Intel Extension for Scikit-learn (intelex)
- **Other Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Dashboard**: Plotly, Streamlit
- **Deployment**: Streamlit Cloud, Docker (optional)

## Installation and Setup

### Prerequisites
- Python 3.9+
- `pip` package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Intel Extension for Scikit-learn (optional but recommended for performance)
```bash
pip install scikit-learn-intelex
```

To enable Intel Extension for Scikit-learn, add the following to the beginning of your Python scripts:
```python
from sklearnex import patch_sklearn
patch_sklearn()
```

### Step 5: Run the Web Application
```bash
streamlit run dropout_app.py
```

### Step 6: Open the Application
After running the above command, open your browser and go to `http://localhost:8501` to view the web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://<your-custom-subdomain>.streamlit.app](https://student-dropout-prediction-app.streamlit.app/))

## Project Workflow

### Data Preprocessing (Week 1)
The first week focuses on cleaning and preparing the dataset. Key steps include:
- Handling missing values.
- Encoding categorical variables.
- Scaling numerical features using `MinMaxScaler`.
- Custom binning of specific features such as age and admission grade.

### Dashboard Creation (Week 2)
In the second week, a dashboard is created to visualize insights from the dataset. The dashboard helps analyze key trends such as:
- Age distribution of students.
- Dropout rates based on different features like gender, nationality, and economic hardship.
- Plots for understanding how various features correlate with dropout likelihood.
- Hypothesis testing to see which feature has the most significant effect on the target class.

The dashboard is built using **Plotly** for interactive visualizations and integrated into **Streamlit** for an accessible user interface.

### Feature Engineering (Week 3)
In week 3, new features are engineered to boost the model's predictive performance:
- `Total Units Enrolled`: Sum of curricular units enrolled in the 1st and 2nd semesters.
- `Approval Rate`: Ratio of approved to enrolled units.
- `Economic Hardship`: A composite feature based on unemployment rate, inflation rate, and GDP.
- Interaction and polynomial features are also created using the scikit-learn polynomial features library.

### Model Training and Evaluation (Week 4)
During week 4, the project focuses on building machine learning models to predict student dropout. The models include:
- Logistic Regression (baseline model).
- XGBoost
- Gradient Boost
- Random Forest etc.
Evaluation metrics include:
- Accuracy, precision, recall, F1-score.


### Hyperparameter Tuning (Week 5)
In week 5, hyperparameter tuning is performed to optimize model performance. Techniques like `GridSearchCV` are used to fine-tune model parameters, such as:
- Regularization strength (`C`) for Logistic Regression.
- Learning rate for XGBoost and random forest etc..

### Deployment (Week 6)
In the final week, the machine learning model and the dashboard are deployed as a web application using **Streamlit**. The app allows users to:
- Input student data for predictions.
- Explore key trends via the dashboard.

## Usage
1. Run the app as described in the setup instructions.
2. Input student information into the web form for predictions.
3. Use the dashboard to explore trends and insights from the dataset.

## Contributing
Feel free to contribute by submitting a pull request or opening an issue.

## License
This project is licensed under the Apache 2.0 License.

---

### Notes:
- Update any project-specific paths or repository details.
- You can add more sections for Docker setup, advanced usage, or API integration if applicable.
