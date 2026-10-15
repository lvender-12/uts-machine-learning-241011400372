# Tugas UTS Machine Learning

## Identitas Mahasiswa

- **Nama:** Muhammad Nur Jagat Arya Damar
- **NIM:** 241011400372
- **Kelas:** 05TPLM005
- **Mata Kuliah:** Machine Learning

## Deskripsi

Tugas UTS ini menerapkan dua algoritma Machine Learning:

1. **Simple Linear Regression** untuk memprediksi penjualan berdasarkan biaya iklan.
2. **Simple Logistic Regression** untuk mengklasifikasikan email menjadi spam atau bukan spam.

## File Proyek

- `tugas_UTS.ipynb` — implementasi algoritma, evaluasi, visualisasi, dan kesimpulan.
- `Ads_Data.csv` — dataset regresi linear.
- `Email_Data.csv` — dataset klasifikasi email.

## Teknologi

- Python
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook
- scikit-learn untuk perbandingan model

## Ringkasan Hasil

### 1. Linear Regression

Model digunakan untuk memprediksi penjualan berdasarkan biaya iklan. Implementasi dilakukan menggunakan persamaan normal dan Gradient Descent dengan beberapa nilai learning rate. Performa model dievaluasi menggunakan MAE, MSE, RMSE, dan R².

Hasil implementasi manual menunjukkan performa yang hampir identik dengan model Linear Regression dari scikit-learn.

### 2. Logistic Regression

Model digunakan untuk mengklasifikasikan email sebagai spam atau bukan spam berdasarkan jumlah tautan dan penggunaan kata promosi.

Hasil evaluasi pada data pengujian:

| Metrik | Hasil |
|---|---:|
| Accuracy | 75% |
| Precision | 100% |
| Recall | 42,9% |
| F1-score | 60% |

Model memiliki precision tinggi, tetapi recall masih rendah sehingga sebagian email spam belum berhasil terdeteksi.

## Cara Menjalankan

1. Instal dependensi:

   ```bash
   python -m pip install numpy pandas matplotlib jupyter scikit-learn
   ```

2. Letakkan file notebook dan kedua dataset dalam direktori yang sama.
3. Buka `tugas_UTS.ipynb` menggunakan Jupyter Notebook.
4. Jalankan seluruh cell dari awal hingga akhir.

## Kesimpulan

Linear Regression dapat digunakan untuk memodelkan hubungan antara biaya iklan dan penjualan. Sementara itu, Logistic Regression mampu mengklasifikasikan email spam, tetapi hasil recall menunjukkan bahwa model masih perlu ditingkatkan agar lebih banyak email spam dapat terdeteksi.
