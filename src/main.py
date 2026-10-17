from pathlib import Path
script = r'''import json
from pathlib import Path

# Buat notebook Jupyter valid di folder tempat script ini dijalankan.
OUT = Path("tugas_UTS.ipynb")

cells = []

def markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })

markdown("""# Tugas UTS: Linear Regression & Logistic Regression
**Mata Kuliah:** Machine Learning

Notebook ini mengerjakan Simple Regression (`Ads_Data.csv`) dan Simple Classification (`Email_Data.csv`).

**Ketentuan:** implementasi model dan metrik utama dibuat manual menggunakan NumPy. Scikit-learn hanya digunakan pada bagian bonus.

**RANDOM_STATE:** 72 (dua digit terakhir NIM).
""")

markdown("""## Persiapan dan import library
Pastikan `Ads_Data.csv` dan `Email_Data.csv` berada satu folder dengan notebook ini.
""")

code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 72
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", None)
plt.style.use("default")

ADS_PATH = Path("Ads_Data.csv")
EMAIL_PATH = Path("Email_Data.csv")

if not ADS_PATH.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {ADS_PATH.resolve()}")
if not EMAIL_PATH.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {EMAIL_PATH.resolve()}")

ads_df = pd.read_csv(ADS_PATH)
email_df = pd.read_csv(EMAIL_PATH)

print("Ads_Data.csv:", ads_df.shape)
display(ads_df.head())
print("Email_Data.csv:", email_df.shape)
display(email_df.head())
""")

markdown("""## Pemeriksaan data
Kolom yang diharapkan:
- Ads: `BiayaIklan`, `Penjualan`
- Email: `JumlahLink`, `KataPromo`, `Spam`
""")

code("""print("Kolom Ads:", ads_df.columns.tolist())
print("Kolom Email:", email_df.columns.tolist())
print("\\nInfo Ads:")
display(ads_df.describe(include="all"))
print("\\nInfo Email:")
display(email_df.describe(include="all"))

required_ads = {"BiayaIklan", "Penjualan"}
required_email = {"JumlahLink", "KataPromo", "Spam"}
if not required_ads.issubset(ads_df.columns):
    raise ValueError(f"Kolom Ads harus mencakup {required_ads}; ditemukan {set(ads_df.columns)}")
if not required_email.issubset(email_df.columns):
    raise ValueError(f"Kolom Email harus mencakup {required_email}; ditemukan {set(email_df.columns)}")
""")

markdown("""# A. Simple Linear Regression

## A1. Memahami dan memvisualisasikan data
""")

code("""X_ads_all = ads_df["BiayaIklan"].to_numpy(dtype=float)
y_ads_all = ads_df["Penjualan"].to_numpy(dtype=float)

plt.figure(figsize=(8, 5))
plt.scatter(X_ads_all, y_ads_all)
plt.xlabel("Biaya Iklan")
plt.ylabel("Penjualan")
plt.title("Hubungan Biaya Iklan dan Penjualan")
plt.grid(True, alpha=0.3)
plt.show()

print("Jumlah data:", len(ads_df))
print("Korelasi:", np.corrcoef(X_ads_all, y_ads_all)[0, 1])
""")

markdown("""## A2. Membagi data train dan test
Pembagian manual menggunakan `RANDOM_STATE=72`, dengan 80% data training dan 20% data testing.
""")

code("""def train_test_split_manual(X, y, test_size=0.2, random_state=72):
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X dan y harus memiliki jumlah baris yang sama")
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = int(np.ceil(len(X) * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train_ads, X_test_ads, y_train_ads, y_test_ads = train_test_split_manual(
    X_ads_all.reshape(-1, 1), y_ads_all, test_size=0.2, random_state=RANDOM_STATE
)
print("Data train:", len(X_train_ads))
print("Data test:", len(X_test_ads))
""")

markdown("""## A3. Model Linear Regression dengan persamaan normal
Model: ŷ = b₀ + b₁x. Parameter dihitung menggunakan least squares.
""")

code("""def fit_linear_regression_normal(X, y):
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X_bias = np.column_stack([np.ones(len(X)), X])
    theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    return theta.ravel()

def predict_linear(X, theta):
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    X_bias = np.column_stack([np.ones(len(X)), X])
    return X_bias @ np.asarray(theta)

theta_normal = fit_linear_regression_normal(X_train_ads, y_train_ads)
pred_train_normal = predict_linear(X_train_ads, theta_normal)
pred_test_normal = predict_linear(X_test_ads, theta_normal)
print(f"Intercept (b0): {theta_normal[0]:.6f}")
print(f"Koefisien (b1): {theta_normal[1]:.6f}")
""")

markdown("""## A4. Evaluasi model
Metrik dihitung manual: MAE, MSE, RMSE, dan R².
""")

code("""def mae_manual(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse_manual(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse_manual(y_true, y_pred):
    return float(np.sqrt(mse_manual(y_true, y_pred)))

def r2_manual(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

linear_results = pd.DataFrame({
    "Metrik": ["MAE", "MSE", "RMSE", "R2"],
    "Train": [
        mae_manual(y_train_ads, pred_train_normal),
        mse_manual(y_train_ads, pred_train_normal),
        rmse_manual(y_train_ads, pred_train_normal),
        r2_manual(y_train_ads, pred_train_normal),
    ],
    "Test": [
        mae_manual(y_test_ads, pred_test_normal),
        mse_manual(y_test_ads, pred_test_normal),
        rmse_manual(y_test_ads, pred_test_normal),
        r2_manual(y_test_ads, pred_test_normal),
    ],
})
display(linear_results)
""")

markdown("""## A5. Gradient Descent dengan beberapa learning rate
Gradient descent dijalankan selama 5000 iterasi dengan learning rate 0.001, 0.02, dan 0.1. Fitur distandardisasi agar optimisasi lebih stabil; prediksi tetap dikonversi ke satuan target asli.
""")

code("""def fit_linear_gradient_descent(X, y, learning_rate=0.001, iterations=5000):
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mean_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x == 0] = 1
    X_scaled = (X - mean_x) / std_x
    X_bias = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    theta = np.zeros(X_bias.shape[1], dtype=float)
    losses = []
    for _ in range(iterations):
        pred = X_bias @ theta
        error = pred - y
        gradient = (2 / len(y)) * (X_bias.T @ error)
        theta -= learning_rate * gradient
        losses.append(np.mean((X_bias @ theta - y) ** 2))
    return theta, mean_x, std_x, np.asarray(losses)

def predict_linear_gd(X, fitted):
    theta, mean_x, std_x, _ = fitted
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    X_scaled = (X - mean_x) / std_x
    return np.column_stack([np.ones(len(X_scaled)), X_scaled]) @ theta

learning_rates = [0.001, 0.02, 0.1]
gd_models = {}
gd_rows = []
for lr in learning_rates:
    fitted = fit_linear_gradient_descent(X_train_ads, y_train_ads, learning_rate=lr, iterations=5000)
    gd_models[lr] = fitted
    pred = predict_linear_gd(X_test_ads, fitted)
    gd_rows.append({
        "Learning rate": lr,
        "MSE train": mse_manual(y_train_ads, predict_linear_gd(X_train_ads, fitted)),
        "MSE test": mse_manual(y_test_ads, pred),
        "Loss akhir": fitted[3][-1],
    })

display(pd.DataFrame(gd_rows))

plt.figure(figsize=(8, 5))
for lr, fitted in gd_models.items():
    plt.plot(fitted[3], label=f"learning rate={lr}")
plt.xlabel("Iterasi")
plt.ylabel("MSE")
plt.title("Konvergensi Gradient Descent")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
""")

markdown("""## A6. Visualisasi hasil prediksi dan kesimpulan
""")

code("""best_lr = min(gd_models, key=lambda lr: mse_manual(y_test_ads, predict_linear_gd(X_test_ads, gd_models[lr])))
best_gd = gd_models[best_lr]
pred_test_gd = predict_linear_gd(X_test_ads, best_gd)

order = np.argsort(X_test_ads[:, 0])
plt.figure(figsize=(8, 5))
plt.scatter(X_train_ads[:, 0], y_train_ads, label="Train", alpha=0.7)
plt.scatter(X_test_ads[:, 0], y_test_ads, label="Test aktual", marker="x")
plt.plot(X_test_ads[order, 0], pred_test_gd[order], label=f"Prediksi GD (lr={best_lr})")
plt.xlabel("Biaya Iklan")
plt.ylabel("Penjualan")
plt.title("Prediksi Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Learning rate dengan MSE test terendah:", best_lr)
print("MSE test:", mse_manual(y_test_ads, pred_test_gd))
print("R2 test:", r2_manual(y_test_ads, pred_test_gd))
""")

markdown("""# B. Simple Logistic Regression

## B1. Memahami data dan distribusi kelas
`Spam` harus berupa label biner (0 dan 1). Fitur: `JumlahLink` dan `KataPromo`.
""")

code("""email = email_df.copy()
for col in ["JumlahLink", "KataPromo", "Spam"]:
    email[col] = pd.to_numeric(email[col], errors="raise")

X_email_all = email[["JumlahLink", "KataPromo"]].to_numpy(dtype=float)
y_email_all = email["Spam"].to_numpy(dtype=float)
classes = np.unique(y_email_all)
if not set(classes).issubset({0.0, 1.0}) or len(classes) != 2:
    raise ValueError(f"Kolom Spam harus berisi dua kelas 0 dan 1; ditemukan {classes}")

display(email["Spam"].value_counts().sort_index().rename("Jumlah").to_frame())
email["Spam"].value_counts().sort_index().plot(kind="bar", rot=0)
plt.xlabel("Kelas Spam")
plt.ylabel("Jumlah")
plt.title("Distribusi Kelas Email")
plt.show()
""")

markdown("""## B2. Membagi data train dan test
Menggunakan 75% data training dan 25% data testing.
""")

code("""X_train_email, X_test_email, y_train_email, y_test_email = train_test_split_manual(
    X_email_all, y_email_all, test_size=0.25, random_state=RANDOM_STATE
)
print("Data train:", len(X_train_email))
print("Data test:", len(X_test_email))
""")

markdown("""## B3. Implementasi Logistic Regression manual dengan Gradient Descent
Learning rate 0.1 dan 10000 iterasi. Fitur distandardisasi untuk stabilitas numerik.
""")

code("""def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def fit_logistic_regression(X, y, learning_rate=0.1, iterations=10000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    mean_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x == 0] = 1
    X_scaled = (X - mean_x) / std_x
    X_bias = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    theta = np.zeros(X_bias.shape[1], dtype=float)
    losses = []
    eps = 1e-12
    for _ in range(iterations):
        probs = sigmoid(X_bias @ theta)
        gradient = (X_bias.T @ (probs - y)) / len(y)
        theta -= learning_rate * gradient
        probs_after = np.clip(sigmoid(X_bias @ theta), eps, 1 - eps)
        loss = -np.mean(y * np.log(probs_after) + (1 - y) * np.log(1 - probs_after))
        losses.append(loss)
    return theta, mean_x, std_x, np.asarray(losses)

def predict_proba_logistic(X, fitted):
    theta, mean_x, std_x, _ = fitted
    X = np.asarray(X, dtype=float)
    X_scaled = (X - mean_x) / std_x
    X_bias = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    return sigmoid(X_bias @ theta)

def predict_logistic(X, fitted, threshold=0.5):
    return (predict_proba_logistic(X, fitted) >= threshold).astype(int)

logistic_model = fit_logistic_regression(
    X_train_email, y_train_email, learning_rate=0.1, iterations=10000
)
plt.figure(figsize=(8, 5))
plt.plot(logistic_model[3])
plt.xlabel("Iterasi")
plt.ylabel("Binary cross-entropy")
plt.title("Loss Logistic Regression")
plt.grid(True, alpha=0.3)
plt.show()
""")

markdown("""## B4. Evaluasi klasifikasi manual
Metrik: confusion matrix, accuracy, precision, recall, F1-score, dan log loss.
""")

code("""def confusion_matrix_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return np.array([
        [np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
        [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))],
    ])

def classification_metrics_manual(y_true, y_pred):
    cm = confusion_matrix_manual(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return cm, {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }

pred_train_email = predict_logistic(X_train_email, logistic_model)
pred_test_email = predict_logistic(X_test_email, logistic_model)
cm_train, metrics_train = classification_metrics_manual(y_train_email, pred_train_email)
cm_test, metrics_test = classification_metrics_manual(y_test_email, pred_test_email)

print("Confusion matrix test (baris aktual, kolom prediksi):")
display(pd.DataFrame(cm_test, index=["Aktual 0", "Aktual 1"], columns=["Prediksi 0", "Prediksi 1"]))
display(pd.DataFrame({"Train": metrics_train, "Test": metrics_test}))

proba_test_email = np.clip(predict_proba_logistic(X_test_email, logistic_model), 1e-12, 1 - 1e-12)
log_loss_test = -np.mean(
    y_test_email * np.log(proba_test_email)
    + (1 - y_test_email) * np.log(1 - proba_test_email)
)
print("Log loss test:", float(log_loss_test))
""")

markdown("""## B5. Pengaruh threshold
Bandingkan threshold 0.3, 0.5, dan 0.7 untuk melihat perubahan precision dan recall.
""")

code("""threshold_rows = []
prob_test = predict_proba_logistic(X_test_email, logistic_model)
for threshold in [0.3, 0.5, 0.7]:
    pred = (prob_test >= threshold).astype(int)
    _, metrics = classification_metrics_manual(y_test_email, pred)
    threshold_rows.append({"Threshold": threshold, **metrics})
display(pd.DataFrame(threshold_rows))
""")

markdown("""## B6. Visualisasi dan kesimpulan
""")

code("""plt.figure(figsize=(7, 5))
plt.scatter(X_test_email[:, 0], X_test_email[:, 1], c=y_test_email, marker="o", label="Aktual", alpha=0.8)
plt.xlabel("Jumlah Link")
plt.ylabel("Kata Promo")
plt.title("Data Email Test (warna menunjukkan kelas aktual)")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Accuracy test: {metrics_test['Accuracy']:.3f}")
print(f"Precision test: {metrics_test['Precision']:.3f}")
print(f"Recall test: {metrics_test['Recall']:.3f}")
print(f"F1-score test: {metrics_test['F1-score']:.3f}")
print("Kesimpulan: metrik di atas menggambarkan kemampuan model membedakan email spam dan bukan spam pada data test.")
""")

markdown("""# C. Pertanyaan Konseptual

## C1. Apa perbedaan Linear Regression dan Logistic Regression?
Linear Regression memprediksi nilai numerik kontinu, sedangkan Logistic Regression memprediksi probabilitas kelas dan digunakan untuk klasifikasi.

## C2. Mengapa data dibagi menjadi training dan testing?
Training digunakan untuk mempelajari parameter model. Testing digunakan untuk mengukur kemampuan model pada data yang tidak digunakan saat pelatihan.

## C3. Apa fungsi learning rate pada Gradient Descent?
Learning rate menentukan besar langkah pembaruan parameter setiap iterasi. Terlalu kecil membuat proses lambat, sedangkan terlalu besar dapat menyebabkan optimisasi tidak stabil atau gagal konvergen.

## C4. Apa itu overfitting?
Overfitting terjadi ketika model terlalu menyesuaikan diri dengan data training sehingga performanya baik pada training tetapi buruk pada data baru.

## C5. Mengapa evaluasi model penting?
Evaluasi membantu mengukur kualitas prediksi, membandingkan model, dan memahami jenis kesalahan yang dibuat sebelum model digunakan.
""")

markdown("""# Bonus. Perbandingan dengan scikit-learn
Bagian ini opsional. Model utama dan metrik sebelumnya tetap diimplementasikan manual menggunakan NumPy.
""")

code("""try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.model_selection import train_test_split as sklearn_split

    Xtr, Xte, ytr, yte = sklearn_split(
        X_ads_all.reshape(-1, 1), y_ads_all,
        test_size=0.2, random_state=RANDOM_STATE
    )
    sk_linear = LinearRegression().fit(Xtr, ytr)
    sk_pred = sk_linear.predict(Xte)
    print("Sklearn LinearRegression")
    print("Intercept:", sk_linear.intercept_)
    print("Koefisien:", sk_linear.coef_[0])
    print("MSE test:", mean_squared_error(yte, sk_pred))

    Xtr_e, Xte_e, ytr_e, yte_e = sklearn_split(
        X_email_all, y_email_all.astype(int),
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_email_all.astype(int)
    )
    sk_logistic = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    sk_logistic.fit(Xtr_e, ytr_e)
    sk_pred_e = sk_logistic.predict(Xte_e)
    print("\\nSklearn LogisticRegression")
    print("Accuracy test:", accuracy_score(yte_e, sk_pred_e))
except ImportError:
    print("Scikit-learn belum terpasang. Untuk menjalankan bagian bonus: pip install scikit-learn")
""")

markdown("""## Selesai
Jalankan semua cell dari atas ke bawah (**Run → Run All**) dan simpan notebook setelah output muncul. Pastikan file CSV ikut dikumpulkan bersama notebook.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "codemirror_mode": {"name": "ipython", "version": 3}
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Validasi JSON notebook, lalu simpan ke file.
json.dumps(notebook, ensure_ascii=False)
# Simpan generator ini sebagai notebook.py; saat dijalankan, output notebook ada di folder kerja.
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Berhasil membuat notebook: {OUT.resolve()}")
print(f"Jumlah cell: {len(cells)}")
'''
path = Path("./src/notebook.py")
path.write_text(script, encoding="utf-8")
print(f"Created {path} ({len(script.splitlines())} lines)")
