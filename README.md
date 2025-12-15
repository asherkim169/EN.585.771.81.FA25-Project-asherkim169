# EN.585.771.81.FA25-Project-asherkim169
EN.585.771.81.FA25 Capstone Project

COVID-19 Case Prediction App

Linear Regression vs Neural Networks (Streamlit)

1. Purpose of the Project

1.1 This project builds an interactive Streamlit application to model and predict COVID-19 confirmed case counts using:
* a distributed lag linear regression (OLS)
* Fully connected neural networks with user-controlled architecture

1.2 The goal is to compare traditional regression with neural networks under different architectural choices and training settings.

2. How to Run the Code
Step 1: Install Required Packages
pip install streamlit pandas numpy statsmodels scikit-learn torch matplotlib
If PyTorch is not installed:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Step 2: Place the Data File
Ensure the following file exists at the path specified in the code:
time_series_covid19_confirmed_global.csv
This file contains global COVID-19 confirmed case counts.

Step 3: Launch the Streamlit App
streamlit run A_Kim_Capstone_Project.py

A browser window will open automatically.

3. How to Use the App

3.1 User Inputs (Left Panel)
3.1.1 Country Name (Text Field)
Enter a country exactly as it appears in the dataset
Example:
-Korea, South
-China
-Italy
3.1.2 Neural Network Architecture
-Select number of hidden layers (1–8)
-Select number of nodes per layer (1–4)
3.1.3 Training Hyperparameters
-Number of epochs
-Learning rate

3.2 Outputs
3.2.1 Prediction plot showing:
-True case counts
-OLS regression predictions
-Neural network predictions
3.2.2 Summary table comparing Mean Squared Error (MSE) across models

If the country name does not match the dataset, a warning is shown.

4. Code Breakdown
4.1 Imports
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import torch

-streamlit: Web app interface
-pandas, numpy: Data handling
-statsmodels: Linear regression (OLS)
-torch: Neural network modeling
-sklearn: Scaling and MSE evaluation

4.2 Data Loading
@st.cache_data
def load_data(path):
    return pd.read_csv(path)
-Loads the CSV once
-Caches data to speed up app reruns

4.3 Time Series Preparation
def prepare_country_data(df, country, k=5):

Steps:
-Filters data for the selected country
-Aggregates provinces (if applicable)
-Transposes data so rows represent days
-Defines day 0 as the first day with >0 cases
-Creates 5 lag variables: 
𝑌t-1 ........ Yt-5

4.4 Distributed Lag Linear Regression
def fit_distributed_lag(ts, k=5):


Implements:
𝑌t = 𝛼0 + 𝛽1Yt-1 + ⋯ + 𝛽5𝑌𝑡−5 + 𝜖𝑡
-Fitted using OLS
-Used as a baseline model
-Produces in-sample predictions and MSE

4.5 Neural Network Architecture
class DynamicNet(nn.Module):
-Fully connected feedforward network
-User-controlled:
--Number of hidden layers
--Nodes per layer
-ReLU activation for hidden layers
-Linear output layer

4.6 Neural Network Training
def train_nn(...)
-Uses Mean Squared Error loss
-Adam optimizer
-Early stopping for efficiency
-Trains on first half of time series
-Evaluates on second half

4.7 Model Comparison
-OLS regression MSE
-Neural network MSE for each configuration
-Results stored in a table and sorted by error

4.8 Visualization
-Line plot comparing:
-True case counts
-OLS predictions
-Neural network predictions
-Helps visually assess overfitting and prediction accuracy

5. Interpretation of Results
-OLS provides a simple linear benchmark
-Neural networks can capture nonlinear dynamics
-Increasing layers and nodes does not always improve performance
-MSE comparison highlights the bias–variance tradeoff

6. Link to video
https://github.com/asherkim169/EN.585.771.81.FA25-Project-asherkim169/blob/main/A_Kim_Capstone_Video.mp4
