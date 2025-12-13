import streamlit as st              # Library untuk membuat web dashboard
import pandas as pd                # Library untuk membaca dan mengolah data
import numpy as np                 # Library untuk perhitungan numerik
import matplotlib.pyplot as plt    # Library untuk visualisasi grafik

st.title("Dashboard Curve Fitting Data Sensor IoT (LSM)")
st.write("Visualisasi hubungan Suhu dan Cahaya terhadap Kelembaban menggunakan Least Squares Method")

# ============================
# MEMBACA DATA SENSOR
# ============================

# Membaca file CSV berisi data sensor
data = pd.read_csv("data_sensor.csv")

# Menampilkan tabel data pada dashboard
st.subheader("Data Sensor")
st.dataframe(data)


# ============================
# PENGAMBILAN VARIABEL
# ============================

x_suhu = data['suhu'].values        # Data suhu (°C)
x_cahaya = data['cahaya'].values    # Data intensitas cahaya (Lux)
y = data['kelembaban'].values       # Data kelembaban (%)


# ============================
# FUNGSI LEAST SQUARES METHOD
# ============================

def lsm(x, y):
    """
    Fungsi untuk menghitung parameter regresi linear
    menggunakan metode Least Squares Method (LSM)

    y = a*x + b
    """
    n = len(x)  # Jumlah data

    # Menghitung koefisien a (slope)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
        (n * np.sum(x**2) - (np.sum(x))**2)

    # Menghitung konstanta b (intercept)
    b = (np.sum(y) - a * np.sum(x)) / n

    return a, b

# ============================
# PERHITUNGAN MODEL REGRESI
# ============================

# Regresi Suhu terhadap Kelembaban
a_suhu, b_suhu = lsm(x_suhu, y)

# Regresi Cahaya terhadap Kelembaban
a_cahaya, b_cahaya = lsm(x_cahaya, y)

# ============================
# MENAMPILKAN MODEL REGRESI
# ============================

st.subheader("Model Regresi Linear (LSM)")

# Menampilkan persamaan regresi suhu
st.write(f"Suhu → Kelembaban : y = {a_suhu:.2f}x + {b_suhu:.2f}")

# Menampilkan persamaan regresi cahaya
st.write(f"Cahaya → Kelembaban : y = {a_cahaya:.4f}x + {b_cahaya:.2f}")

# ============================
# VISUALISASI GRAFIK SUHU
# ============================

# Menghitung nilai prediksi kelembaban berdasarkan suhu
y_pred_suhu = a_suhu * x_suhu + b_suhu

# Membuat grafik
fig1, ax1 = plt.subplots()
ax1.scatter(x_suhu, y, label="Data Sensor")
ax1.plot(x_suhu, y_pred_suhu, label="Garis Regresi LSM")
ax1.set_xlabel("Suhu (°C)")
ax1.set_ylabel("Kelembaban (%)")
ax1.set_title("Curve Fitting Suhu vs Kelembaban")
ax1.legend()

# Menampilkan grafik di Streamlit
st.pyplot(fig1)

# ============================
# VISUALISASI GRAFIK CAHAYA
# ============================

# Menghitung nilai prediksi kelembaban berdasarkan cahaya
y_pred_cahaya = a_cahaya * x_cahaya + b_cahaya

# Membuat grafik
fig2, ax2 = plt.subplots()
ax2.scatter(x_cahaya, y, label="Data Sensor")
ax2.plot(x_cahaya, y_pred_cahaya, label="Garis Regresi LSM")
ax2.set_xlabel("Cahaya (Lux)")
ax2.set_ylabel("Kelembaban (%)")
ax2.set_title("Curve Fitting Cahaya vs Kelembaban")
ax2.legend()

# Menampilkan grafik di Streamlit
st.pyplot(fig2)
