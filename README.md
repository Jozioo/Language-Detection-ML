# Language Detection - Machine Learning

> **Proyek NLP — Kelas LK01 | Semester Genap 2025/2026**  
> 2802429900 Christian Leonardo Halim · 2802426060 Darren Christian Pramana · 2802429863 Jozio Damaier Gidalti

## Struktur Project

```
project/
│
├── Language_Detection_NLP.ipynb   # Notebook utama
├── dataset.csv                    # Dataset multibahasa
├── README.md                      # Dokumentasi ini
├── app.py                         # Aplikasi demo menggunakan streamlit
│
└── saved_models/                  # Folder output model
    ├── mnb_language_detector.pkl          # Model utama (MNB)
    ├── logistic_regression_pipeline.pkl   # Pembanding 1
    ├── linear_svm_pipeline.pkl            # Pembanding 2
    ├── fasttext_style_pipeline.pkl        # Pembanding 3
    └── knn_pipeline.pkl                   # Pembanding 4
```

## ⚙️ Persyaratan Sistem

### Python Version
```
Python >= 3.8
```

### Instalasi Dependensi
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib notebook
```

Atau menggunakan `requirements.txt`:
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.1.0
notebook>=6.4.0
```
