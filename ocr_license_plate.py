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

def get_license_plate_number(image_path, model="llava-v1.5-7b"):
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
                        "text": "What are the Indonesian license plate numbers shown in this image? Respond with each plate number on a new line."
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
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(LMSTUDIO_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Pisahkan prediksi ke dalam daftar nomor plat
        predictions = result['choices'][0]['message']['content'].strip().split('\n')
        return [pred.strip() for pred in predictions if pred.strip()]
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []

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

def read_labels(label_path):
    """Membaca semua nomor plat dari file label"""
    ground_truths = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 6:  # Minimal 5 angka (YOLO) + nomor plat
                        ground_truths.append(parts[-1])  # Nomor plat adalah token terakhir
                    else:
                        print(f"Invalid format in {label_path}: {line}")
        return ground_truths
    except Exception as e:
        print(f"Error reading {label_path}: {str(e)}")
        return []

def match_predictions(ground_truths, predictions):
    """Mencocokkan prediksi dengan ground truth berdasarkan CER"""
    results = []
    used_predictions = set()
    
    for gt in ground_truths:
        best_cer = float('inf')
        best_pred = ""
        
        for pred in predictions:
            if pred not in used_predictions:
                cer = calculate_cer(gt, pred)
                if cer < best_cer:
                    best_cer = cer
                    best_pred = pred
        
        results.append((gt, best_pred, best_cer))
        if best_pred:
            used_predictions.add(best_pred)
    
    # Tambahkan ground truth yang tidak memiliki prediksi
    for gt in ground_truths:
        if not any(r[0] == gt for r in results):
            results.append((gt, "", 1.0))
    
    return results

def main():
    # Direktori dataset
    dataset_dir = "C:\Users\Kevin Tobing\.lmstudio\models\Indonesian License Plate Dataset"  # Ganti dengan path lokal dataset
    output_csv = "ocr_results.csv"
    
    # List untuk menyimpan hasil
    results = []
    
    # Subfolder untuk diproses
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labelswithLP', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {split}: Directory not found")
            continue
        
        # Proses setiap gambar
        for image_name in os.listdir(images_dir):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, image_name)
                
                # Mendapatkan path label (ganti ekstensi ke .txt)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                # Membaca semua ground truth
                ground_truths = read_labels(label_path) if os.path.exists(label_path) else []
                
                # Mendapatkan prediksi dari model
                predictions = get_license_plate_number(image_path)
                
                # Mencocokkan prediksi dengan ground truth
                matched_results = match_predictions(ground_truths, predictions)
                
                # Menyimpan hasil
                for gt, pred, cer_score in matched_results:
                    results.append({
                        "image": f"{split}/{image_name}",
                        "ground_truth": gt,
                        "prediction": pred,
                        "CER_score": cer_score
                    })
    
    # Menyimpan hasil ke CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image', 'ground_truth', 'prediction', 'CER_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Hasil telah disimpan ke {output_csv}")

if __name__ == "__main__":
    main()
