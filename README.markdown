OCR License Plate Recognition with VLM and LMStudio
Deskripsi
Program ini melakukan Optical Character Recognition (OCR) untuk mengenali nomor plat kendaraan dari gambar menggunakan Visual Language Model (VLM) melalui LMStudio dan diintegrasikan dengan Python. Dataset yang digunakan adalah Indonesian License Plate Dataset dari Kaggle. Hasil prediksi disimpan dalam format CSV dengan metrik evaluasi Character Error Rate (CER).
Prasyarat

Python 3.8+
LMStudio terinstal dan berjalan di http://localhost:1234
Model VLM (misalnya, LLaVA v1.5-7B) telah dimuat di LMStudio
Indonesian License Plate Dataset diunduh dan diekstrak

Instalasi

Clone repository ini:git clone <your-repo-url>
cd <your-repo-directory>


Install dependensi:pip install requests pillow python-Levenshtein


Unduh dataset dari Kaggle dan ekstrak ke folder lokal, misalnya Indonesian License Plate Dataset/.
Pastikan LMStudio berjalan dengan model VLM (contoh: llava-v1.5-7b).

Struktur Dataset
Dataset harus memiliki struktur seperti berikut:
Indonesian License Plate Dataset/
├── images/
│   ├── test/
│   │   ├── test001.jpg
│   │   └── ...
│   ├── train/
│   │   ├── train001.jpg
│   │   └── ...
│   ├── val/
│   │   ├── val001.jpg
│   │   └── ...
├── labels/
│   ├── test/
│   ├── train/
│   ├── val/
├── labelswithLP/
│   ├── test/
│   │   ├── test001.txt
│   │   └── ...
│   ├── train/
│   │   ├── train001.txt
│   │   └── ...
│   ├── val/
│   │   ├── val001.txt
│   │   └── ...


Folder images berisi gambar plat nomor (format .jpg, .jpeg, atau .png).
Folder labelswithLP berisi file teks (.txt) dengan nama sesuai gambar. Setiap file berisi satu atau lebih baris anotasi YOLO (class_id, x_center, y_center, width, height, nomor_plat). Contoh:0 0.23545 0.771991 0.074074 0.022156 B9140BCD
0 0.472388 0.717593 0.050595 0.017857 B2407UZO
0 0.780423 0.7333 0.062831 0.020833 B2842PKM


Folder labels berisi anotasi bounding box tanpa nomor plat (tidak digunakan dalam kode ini).

Cara Menjalankan

Sesuaikan dataset_dir di ocr_license_plate.py dengan path dataset Anda, misalnya:dataset_dir = "C:/Users/YourName/Documents/Indonesian License Plate Dataset"


Jalankan program:python ocr_license_plate.py


Hasil akan disimpan di ocr_results.csv dengan kolom: image, ground_truth, prediction, CER_score. Setiap nomor plat dari satu gambar memiliki baris sendiri.

Metrik Evaluasi

Character Error Rate (CER) dihitung dengan rumus:CER = (S + D + I) / N

di mana:
S: Jumlah karakter salah substitusi
D: Jumlah karakter yang dihapus
I: Jumlah karakter yang disisipkan
N: Jumlah karakter pada ground truth



Catatan

Pastikan LMStudio berjalan sebelum menjalankan program.
Sesuaikan LMSTUDIO_API_URL dan LMSTUDIO_API_KEY jika diperlukan.
Gunakan gambar dengan kualitas baik untuk hasil OCR yang optimal.
Dataset tidak disertakan di repository ini; unduh dari Kaggle.
Kode menangani beberapa nomor plat per gambar dengan mencocokkan prediksi berdasarkan CER.
Jika VLM tidak mendeteksi semua nomor plat, coba gunakan model dengan kapasitas lebih tinggi (misalnya, llava-v1.6-13b) atau sesuaikan prompt.

Lisensi
MIT License
