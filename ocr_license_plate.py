import os
import csv
import requests
import json
from PIL import Image
import base64
from pathlib import Path
import Levenshtein

# Konfigurasi LMStudio API
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_API_KEY = "lmstudio-api-key"  # Ganti jika diperlukan

def encode_image(image_path):
    """Mengkodekan gambar ke format base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_license_plate_number(image_path, model="llava"):
    """Mengirim request ke LMStudio untuk mendapatkan nomor plat"""
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the license plate number shown in this image? Respond only with the plate number."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(LMSTUDIO_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return ""

def calculate_cer(ground_truth, prediction):
    """Menghitung Character Error Rate (CER)"""
    S = Levenshtein.substitutions(ground_truth, prediction)
    D = Levenshtein.deletions(ground_truth, prediction)
    I = Levenshtein.insertions(ground_truth, prediction)
    N = len(ground_truth)
    
    if N == 0:
        return 1.0 if prediction else 0.0
    
    cer = (S + D + I) / N
    return cer

def main():
    # Direktori dataset
    dataset_dir = "path/to/your/dataset"  # Ganti dengan path dataset Anda
    output_csv = "ocr_results.csv"
    
    # List untuk menyimpan hasil
    results = []
    
    # Asumsi dataset memiliki struktur: dataset_dir/images/ dan dataset_dir/labels.txt
    # labels.txt berisi: nama_file ground_truth
    labels_file = os.path.join(dataset_dir, "labels.txt")
    images_dir = os.path.join(dataset_dir, "images")
    
    # Membaca ground truth dari labels.txt
    ground_truths = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            for line in f:
                image_name, gt = line.strip().split()
                ground_truths[image_name] = gt
    
    # Proses setiap gambar
    for image_name in os.listdir(images_dir):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, image_name)
            
            # Mendapatkan prediksi dari model
            prediction = get_license_plate_number(image_path)
            
            # Mendapatkan ground truth
            ground_truth = ground_truths.get(image_name, "")
            
            # Menghitung CER
            cer_score = calculate_cer(ground_truth, prediction)
            
            # Menyimpan hasil
            results.append({
                "image": image_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "CER_score": cer_score
            })
    
    # Menyimpan hasil ke CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image', 'ground_truth', 'prediction', 'CER_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Hasil telah disimpan ke {output_csv}")

if __name__ == "__main__":
    main()