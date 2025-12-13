import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_sensor.csv")

x_suhu = data['suhu'].values
x_cahaya = data['cahaya'].values
y = data['kelembaban'].values

def lsm(x, y):
    n = len(x)
    a = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
    b = (np.sum(y) - a*np.sum(x)) / n
    return a, b

a_suhu, b_suhu = lsm(x_suhu, y)
a_cahaya, b_cahaya = lsm(x_cahaya, y)

st.title("Dashboard Curve Fitting Data Sensor IoT (LSM)")

st.subheader("Model Regresi")
st.write(f"Suhu → Kelembaban : y = {a_suhu:.2f}x + {b_suhu:.2f}")
st.write(f"Cahaya → Kelembaban : y = {a_cahaya:.4f}x + {b_cahaya:.2f}")

fig1, ax1 = plt.subplots()
ax1.scatter(x_suhu, y)
ax1.plot(x_suhu, a_suhu*x_suhu + b_suhu)
ax1.set_xlabel("Suhu (°C)")
ax1.set_ylabel("Kelembaban (%)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(x_cahaya, y)
ax2.plot(x_cahaya, a_cahaya*x_cahaya + b_cahaya)
ax2.set_xlabel("Cahaya (Lux)")
ax2.set_ylabel("Kelembaban (%)")
st.pyplot(fig2)
