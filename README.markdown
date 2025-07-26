# OCR License Plate Recognition with VLM and LMStudio

## Deskripsi
Program ini melakukan Optical Character Recognition (OCR) untuk mengenali nomor plat kendaraan dari gambar menggunakan Visual Language Model (VLM) melalui LMStudio dan diintegrasikan dengan Python. Hasil prediksi disimpan dalam format CSV dengan metrik evaluasi Character Error Rate (CER).

## Prasyarat
- Python 3.8+
- LMStudio terinstal dan berjalan di `http://localhost:1234`
- Model VLM (misalnya, LLaVA) telah dimuat di LMStudio
- Dataset gambar plat nomor dan file `labels.txt` dengan format: `nama_file ground_truth`

## Instalasi
1. Clone repository ini:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. Install dependensi:
   ```bash
   pip install requests pillow python-Levenshtein
   ```
3. Pastikan LMStudio berjalan dengan model VLM yang sesuai.

## Struktur Dataset
- Direktori dataset harus memiliki struktur:
  ```
  dataset/
  ├── images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── labels.txt
  ```
- File `labels.txt` berisi pasangan nama file dan ground truth, contoh:
  ```
  image1.jpg AB123CD
  image2.jpg XY789ZZ
  ```

## Cara Menjalankan
1. Sesuaikan `dataset_dir` di `ocr_license_plate.py` dengan path dataset Anda.
2. Jalankan program:
   ```bash
   python ocr_license_plate.py
   ```
3. Hasil akan disimpan di `ocr_results.csv` dengan kolom: `image`, `ground_truth`, `prediction`, `CER_score`.

## Metrik Evaluasi
- **Character Error Rate (CER)** dihitung dengan rumus:
  ```
  CER = (S + D + I) / N
  ```
  di mana:
  - S: Jumlah karakter salah substitusi
  - D: Jumlah karakter yang dihapus
  - I: Jumlah karakter yang disisipkan
  - N: Jumlah karakter pada ground truth

## Catatan
- Pastikan LMStudio berjalan sebelum menjalankan program.
- Sesuaikan `LMSTUDIO_API_URL` dan `LMSTUDIO_API_KEY` jika diperlukan.
- Gunakan gambar dengan kualitas baik untuk hasil OCR yang optimal.

## Lisensi
MIT License