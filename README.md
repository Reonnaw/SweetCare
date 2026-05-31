## SweetCare - Kalkulator Risiko Diabetes
Aplikasi desktop untuk prediksi risiko diabetes menggunakan algoritma Naive Bayes dengan Laplace Smoothing, dibangun dengan Python dan Tkinter.

## Deskripsi
SweetCare adalah aplikasi kalkulator risiko diabetes berbasis GUI yang menerapkan metode klasifikasi Naive Bayes. Aplikasi ini menampilkan langkah-langkah perhitungan secara lengkap dan transparan, cocok untuk keperluan pembelajaran statistika dan machine learning.
- Studi: 4 - Risiko Diabetes
- Dataset: 20 data training
- Metode: Naive Bayes + Laplace Smoothing (α = 1)

## Fitur
- Prediksi risiko diabetes berdasarkan 4 atribut pasien
- Tabel  data latih dengan pewarnaan berdasarkan kelas
- Tampilan langkah perhitungan Naive Bayes secara rinci
- Antarmuka GUI modern dengan tema merah muda (Tkinter)
- Tombol reset untuk mengulang prediksi

## Atribut yang Digunakan 
| Atribut       | Nilai yang Tersedia    |
| ------------- | ---------------------  |
| Umur          | Muda, Paruh Baya, Tua  |
| BMI           | Normal, Tinggi         |
| Gula Darah    | Normal, Tinggi         |
| Tekanan Darah | Normal, Sedang, Tinggi |

## Cara Menjalankan 
**Prasyarat**

Pastikan Python sudah terinstal (versi 3.8 atau lebih baru)
```
python --version
```

Tkinter sudah termasuk dalam instalasi Python standar. Jika belum tersedia:
```
# Ubuntu/Debian

sudo apt-get install python3-tk

# Windows/macOS
# Sudah tersedia secara default
```

Instalasi & Menjalankan
```
# 1. Clone repositori ini
git clone https://github.com/Reonnaw/SweetCare.git

# 2. Masuk ke direktori proyek
cd SweetCare

# 3. (Opsional) Buat dan aktifkan virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 4. Jalankan aplikasi
python main.py
```

## Tampilan Aplikasi
| Panel                             | Keterangan                           | 
| --------------------------------- | ------------------------------------ |
| Data Latih (kiri)                 | Menampilkan 20 sampel training set   | 
| Data Uji (kanan atas)             | Form input atribut pasien            |
| Langkah Perhitungan (kanan bawah) | Output rincian kalkulasi Naive Bayes |

## Metode: Naive Bayes + Laplace Smoothing 
Rumus likelihood yang digunakan:

$$
P(\text{fitur} = v \mid \text{kelas}) = \frac{\text{count}(v, \text{kelas}) + \alpha}{\text{count}(\text{kelas}) + \alpha \times |V|}
$$

Dimana:
α = nilai Laplace smoothing (α = 1)
|v| = jumlah nilai unik pada fitur tersebut

Prediksi dilakukan dengan memilih kelas yang memiliki skor posterior tertinggi:

$$
\hat{y} = \arg\max_{c} \; P(c) \prod_{i} P(x_i \mid c)
$$

## Struktur Proyek

```
SweetCare/
│
├── main.py          # File utama aplikasi
├── README.md        # Dokumentasi proyek
└── .idea/           # Konfigurasi PyCharm (tidak diperlukan untuk menjalankan)
```
