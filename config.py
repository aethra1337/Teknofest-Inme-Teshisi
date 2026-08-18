"""
TEKNOFEST 2025 - İnme Teşhisi ve Zamansal Sınıflandırma
Yapılandırma Dosyası
"""

import os
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent

# Veri seti yolları
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT
ISKEMI_DIR = RAW_DATA_DIR / "iskemi" / "iskemi" / "İskemi Veri Seti"
INMEYOK_DIR = RAW_DATA_DIR / "inmeyok" / "İnme Yok_kronik süreç_diğer Veri Set_PNG" / "İnme Yok_kronik süreç_diğer Veri Set_PNG"

# İşlenmiş veri yolları
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model yolları
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Sonuç yolları
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Model hiperparametreleri
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),  # (height, width, channels)
    "num_classes": 2,  # Hiperakut, Akut, Subakut, Kronik, İnme Yok
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.0001,
    "dropout_rate": 0.7,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
}

# Veri bölümleme
DATA_SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
    "random_seed": 42,
}

# Görüntü ön işleme
IMAGE_PREPROCESSING = {
    "target_size": (224, 224),
    "normalization": "imagenet",  # 'imagenet', 'custom', None
    "augmentation": {
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
        "zoom_range": 0.1,
        "horizontal_flip": True,
        "fill_mode": "constant",
        "cval": 0.0,
    },
}

# Sınıf isimleri
CLASS_NAMES = [
    "Hiperakut",
    "Akut",
    "Subakut",
    "Kronik",
    "İnme Yok"
]

CLASS_NAMES_EN = [
    "Hyperacute",
    "Acute",
    "Subacute",
    "Chronic",
    "No Stroke"
]

# Değerlendirme metrikleri
EVALUATION_METRICS = {
    "primary_metric": "f1_score",  # TEKNOFEST gereksinimi
    "metrics": ["accuracy", "precision", "recall", "f1_score", "auc"],
    "average": "weighted",  # weighted, macro, micro
}

# Transfer Learning (opsiyonel)
TRANSFER_LEARNING = {
    "use_transfer_learning": True,
    "base_model": "EfficientNetB3",  # EfficientNetB3, ResNet50, VGG16, DenseNet121
    "trainable_layers": -1,  # -1: tüm katmanlar, 0: sadece classifier
    "weights": "imagenet",
}

# DICOM ayarları
DICOM_CONFIG = {
    "window_center": None,  # None: otomatik
    "window_width": None,  # None: otomatik
    "rescale_slope": 1.0,
    "rescale_intercept": 0.0,
}

# Loglama
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "training.log",
}

# GPU ayarları
GPU_CONFIG = {
    "use_gpu": True,
    "gpu_memory_growth": True,
    "mixed_precision": False,
}

# Klasörleri oluştur
def create_directories():
    """Gerekli klasörleri oluşturur"""
    directories = [
        PROCESSED_DATA_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CHECKPOINTS_DIR,
        SAVED_MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        PREDICTIONS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Klasör oluşturuldu: {directory}")

if __name__ == "__main__":
    create_directories()

