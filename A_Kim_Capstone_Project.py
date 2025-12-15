import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')

# Data Processing
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

path = r"C:\Users\soo18\OneDrive\Desktop\Data Science\EN.585.771.81.FA25-Project-asherkim169\time_series_covid19_confirmed_global.csv"
df = load_data(path)

def prepare_country_data(df, country, k=5):
    if country not in df['Country/Region'].unique():
        return None
    country_df = df[df['Country/Region'] == country].drop(columns=['Province/State', 'Lat', 'Long'], errors='ignore')
    country_df = country_df.groupby('Country/Region').sum()
    ts = country_df.T
    ts.index = pd.to_datetime(ts.index)
    ts.columns = ['cases']
    first_nonzero_idx = (ts['cases'] > 0).idxmax()
    ts = ts.loc[first_nonzero_idx:].copy()
    ts['t'] = np.arange(len(ts))
    for lag in range(1, k+1):
        ts[f'Y_lag{lag}'] = ts['cases'].shift(lag)
    ts = ts.dropna().reset_index(drop=True)
    return ts


def fit_distributed_lag(ts, k=5):
    X = ts[[f'Y_lag{i}' for i in range(1, k+1)]]
    X = sm.add_constant(X)
    y = ts['cases']
    model = sm.OLS(y, X).fit()
    ts['y_pred'] = model.predict(X)
    ts['error'] = y - ts['y_pred']
    return model, ts


# Nodes
class DynamicNet(nn.Module):
    def __init__(self, input_dim, n_layers=1, n_nodes=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, n_nodes))
        layers.append(nn.ReLU())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_nodes, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_nn(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.02, patience=15):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    best_preds = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            test_loss = criterion(preds, y_test).item()

        if test_loss < best_loss:
            best_loss = test_loss
            best_preds = preds.clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_loss, best_preds


# Streamlit UI
st.title("COVID-19 NN Regression with Flexible Layers & Nodes")

country_input = st.text_input("Enter Country Name:", "Korea, South")
layer_options = st.multiselect("Select number of hidden layers (1–8):", list(range(1, 9)), default=[1,2])
node_options = st.multiselect("Select number of nodes per layer (1–4):", list(range(1, 5)), default=[2,4])
epochs = st.slider("Number of training epochs", 50, 500, 100, step=50)
learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.02, step=0.001)

if country_input in df['Country/Region'].unique():
    ts_data = prepare_country_data(df, country_input)
    k = 5
    A_all_model, A_all_pred = fit_distributed_lag(ts_data, k)
    mse_ols = mean_squared_error(A_all_pred['cases'], A_all_pred['y_pred'])

    split = len(ts_data) // 2
    train_A = ts_data.iloc[:split].copy()
    test_A = ts_data.iloc[split:].copy()

    X_cols = [f'Y_lag{i}' for i in range(1, k+1)]
    X_train_np = train_A[X_cols].values
    y_train_np = train_A['cases'].values.reshape(-1,1)
    X_test_np = test_A[X_cols].values
    y_test_np = test_A['cases'].values.reshape(-1,1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_np)
    y_train_scaled = scaler_y.fit_transform(y_train_np)
    X_test_scaled = scaler_X.transform(X_test_np)
    y_test_scaled = scaler_y.transform(y_test_np)

    X_train_t = torch.from_numpy(X_train_scaled).float()
    y_train_t = torch.from_numpy(y_train_scaled).float()
    X_test_t = torch.from_numpy(X_test_scaled).float()
    y_test_t = torch.from_numpy(y_test_scaled).float()

    results = {"Model": ["OLS"], "MSE": [mse_ols]}
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(y_test_np, label="True Cases", color='black')

    for layers in layer_options:
        for nodes in node_options:
            nn_model = DynamicNet(X_train_t.shape[1], n_layers=layers, n_nodes=nodes)
            mse_nn, preds_nn = train_nn(nn_model, X_train_t, y_train_t, X_test_t, y_test_t,
                                       epochs=epochs, lr=learning_rate)
            preds_denorm = scaler_y.inverse_transform(preds_nn.detach().numpy())
            results["Model"].append(f"NN {layers}L x {nodes}N")
            results["MSE"].append(mse_nn)
            ax.plot(preds_denorm, label=f"NN {layers}L x {nodes}N", linestyle='--')

    ax.plot(test_A['y_pred'].values, label="OLS prediction", linestyle='-.')
    ax.set_title(f"{country_input} COVID Cases Prediction (Test Period)")
    ax.set_xlabel("Days (test period)")
    ax.set_ylabel("Cases")
    ax.legend()
    st.pyplot(fig)

    st.write("### Model Error Comparison")
    st.table(pd.DataFrame(results).sort_values(by='MSE'))
else:
    st.warning("Country not found. Please check spelling or choose another country.")
