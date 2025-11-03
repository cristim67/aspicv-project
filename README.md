# Facial Expression Recognition System

Sistem de recunoaștere a expresiilor faciale pentru proiectul ASPICV.

## Descriere

Acest sistem utilizează metode multiple de extragere a caracteristicilor (HOG, LBP, raw pixels) și algoritmi de clasificare (SVM, MLP, Random Forest) pentru a recunoaște expresiile faciale din imagini alb-negru de 48×48 pixeli.

### Emoții recunoscute
- happy (fericit)
- sad (trist)
- angry (furios)
- neutral
- fear (frică)
- surprise (surpriză)
- disgust (dezgust)

## Structura proiectului

```
aspicv-project/
├── src/                      # Cod sursă principal
│   ├── __init__.py
│   ├── main.py              # Script principal
│   ├── data_loader.py       # Încărcare date
│   ├── feature_extraction.py # Extragere caracteristici
│   └── classifier.py        # Modele de clasificare
├── utils/                    # Utilitare
│   ├── __init__.py
│   └── logger.py            # Configurare logging
├── dataset/                  # Imagini (48×48 pixeli, 1.jpg - 3000.jpg)
├── tags/
│   └── train.csv            # Etichete pentru imagini 1-2700
├── logs/                     # Loguri (generat automat)
├── requirements.txt          # Dependențe Python
└── README.md                # Acest fișier
```

## Instalare

1. Instalați dependențele:
```bash
pip install -r requirements.txt
```

## Utilizare

### Antrenare și predicție completă

Pentru a antrena modelul pe setul de antrenare complet și a genera predicțiile pentru test:

```bash
python src/main.py
```

Scriptul va:
1. Încărca imaginile de antrenare (1-2700)
2. Extrage caracteristici (HOG, LBP, raw pixels)
3. Antrenează un ensemble de clasificatori (SVM + MLP + Random Forest)
4. Generează predicții pentru imagini de test (2701-3000)
5. Creează fișierul `submission.csv`

### Caching (Cache pentru modele)

Sistemul suportă caching automat pentru a evita re-antrenarea:
- **Prima rulare**: antrenează și salvează modelul în `models/`
- **Rulări ulterioare**: încarcă modelul din cache (mult mai rapid!)

```bash
# Cu cache (implicit)
python src/main.py

# Fără cache (forțează re-antrenare)
python src/main.py --no-cache
```

Modelele sunt salvate în folderul `models/`:
- `models/feature_extractor.joblib` - extractorul de caracteristici cu scaler
- `models/classifier.joblib` - modelul antrenat

### Validare (opțional)

Pentru a testa modelul pe o parte din datele de antrenare, modificați în `src/main.py`:
```python
USE_VALIDATION = True  # La linia 45
```

## Rezultate

Fișierul `submission.csv` va avea formatul:
```csv
id,label
2701,happy
2702,angry
...
3000,neutral
```

## Dependențe principale

- `numpy` - Calcule numerice
- `pandas` - Manipulare date
- `opencv-python` - Procesare imagini
- `scikit-learn` - Machine learning
- `scikit-image` - Extragere caracteristici (HOG, LBP)
- `tqdm` - Progress bars

## Logging

Logurile sunt salvate în:
- Console (stdout)
- Fișier: `logs/training.log`

Nivelul de logging poate fi modificat în `src/main.py`:
```python
logger = setup_logger(name="aspicv", log_level=logging.INFO, ...)
```

## Note

- Imagile trebuie să fie în folderul `dataset/`
- Fișierul `tags/train.csv` trebuie să existe
- Setul de test este format din imaginile 2701-3000 (300 imagini)
