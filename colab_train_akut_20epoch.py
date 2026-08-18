"""
🚀 TEKNOFEST İNME TEŞHİSİ - AKUT SINIFI İYİLEŞTİRME
Google Colab + Google Drive için optimize edilmiş eğitim scripti
20 Epoch, GPU ile çalıştırma - SADECE AKUT SINIFI
"""

print("=" * 80)
print("🚀 TEKNOFEST İNME TEŞHİSİ - AKUT SINIFI İYİLEŞTİRME (20 EPOCH)")
print("=" * 80)

# ============================================================================
# ADIM 1: Google Drive Bağlantısı
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 1: Google Drive Bağlantısı")
print("=" * 80)

from google.colab import drive
import os
from pathlib import Path

# Google Drive'ı bağla
drive.mount('/content/drive')
print("✅ Google Drive bağlandı")

# Proje dizinini belirle (Drive'daki proje klasörünüzün yolunu buraya yazın)
# Örnek: /content/drive/MyDrive/Teknofest
PROJECT_PATH = "/content/drive/MyDrive/Teknofest"  # ⚠️ BURAYI KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!

if not os.path.exists(PROJECT_PATH):
    print(f"⚠️ Proje klasörü bulunamadı: {PROJECT_PATH}")
    print("📁 Drive'daki klasörleri kontrol ediyorum...")
    !ls -la /content/drive/MyDrive/
    print("\n⚠️ Yukarıdaki PROJECT_PATH değişkenini kendi proje yolunuza göre güncelleyin!")
else:
    os.chdir(PROJECT_PATH)
    print(f"✅ Proje dizini: {os.getcwd()}")

# ============================================================================
# ADIM 2: Kütüphaneleri Kur
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 2: Kütüphaneler Kuruluyor...")
print("=" * 80)

print("📚 Kütüphaneler kuruluyor...")

# NumPy uyumsuzluğu düzelt
!pip uninstall -y numpy
!pip install -q "numpy<1.26"

# TensorFlow ve diğer kütüphaneler
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kütüphaneler kuruldu!")

# ============================================================================
# ADIM 3: GPU Kontrolü
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 3: GPU Kontrolü")
print("=" * 80)

import tensorflow as tf
import numpy as np

print(f"TensorFlow versiyonu: {tf.__version__}")
print(f"NumPy versiyonu: {np.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i+1}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU bellek büyümesi aktif")
    except RuntimeError as e:
        print(f"⚠️ GPU ayarı hatası: {e}")
else:
    print("❌ GPU bulunamadı! Runtime → Change runtime type → GPU seçin")
    print("⚠️ CPU ile çalışacak, çok yavaş olabilir!")

# ============================================================================
# ADIM 4: Proje Modüllerini İçe Aktar
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 4: Proje Modülleri Yükleniyor...")
print("=" * 80)

import sys
sys.path.insert(0, PROJECT_PATH)

import config
from utils.data_loader import DataLoader
from models.model_builder import build_model
from evaluation.evaluator import ModelEvaluator
from tensorflow import keras
from datetime import datetime
import json
import logging

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Klasörleri oluştur
config.create_directories()

# ============================================================================
# ADIM 5: Veri Setini Hazırla
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 5: Veri Seti Hazırlanıyor...")
print("=" * 80)

data_loader = DataLoader(
    iskemi_dir=config.ISKEMI_DIR,
    inmeyok_dir=config.INMEYOK_DIR,
    processed_dir=config.PROCESSED_DATA_DIR,
)

# Görüntü yollarını topla
print("📦 Görüntü yolları toplanıyor...")
df = data_loader.collect_image_paths()

# Sadece PNG dosyalarını kullan
df = df[df['image_path'].apply(lambda x: str(x).lower().endswith('.png'))]
print(f"✅ Toplam {len(df)} PNG görüntü bulundu")

# SADECE AKUT VE İNME YOK SINIFLARINI KULLAN
print("\n🔍 Sınıf filtreleme: Sadece Akut ve İnme Yok")
df_filtered = df[df['label'].isin(['Akut', 'İnme Yok'])]
print(f"✅ Filtrelenmiş görüntü sayısı: {len(df_filtered)}")
print(f"📊 Sınıf dağılımı:")
print(df_filtered['label'].value_counts())

# Veri setini böl
print("\n📊 Veri seti bölünüyor...")
train_df, val_df, test_df = data_loader.split_data(
    df_filtered,  # Filtrelenmiş veri
    train_ratio=config.DATA_SPLIT["train"],
    val_ratio=config.DATA_SPLIT["val"],
    test_ratio=config.DATA_SPLIT["test"],
    random_seed=config.DATA_SPLIT["random_seed"],
)

print(f"✅ Veri bölümleme:")
print(f"  Eğitim: {len(train_df)} görüntü")
print(f"  Doğrulama: {len(val_df)} görüntü")
print(f"  Test: {len(test_df)} görüntü")

# Sınıf dağılımını göster
print("\n📊 Sınıf dağılımı (Eğitim):")
print(train_df['label'].value_counts())

# İşlenmiş verileri organize et
print("\n📁 Görüntüler kopyalanıyor...")
data_loader.prepare_processed_data(
    train_df,
    val_df,
    test_df,
    config.PROCESSED_DATA_DIR,
    copy_images=True,
)
print("✅ Veri seti hazır!")

# ============================================================================
# ADIM 6: Geliştirilmiş Sınıf Ağırlıkları (AKUT İYİLEŞTİRME)
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 6: Sınıf Ağırlıkları Hesaplanıyor (Akut İyileştirme)")
print("=" * 80)

# Sınıf sayılarını hesapla
class_counts = train_df['label'].value_counts()
total_samples = len(train_df)

# Akut sınıfı için özel ağırlık artırımı
class_weights = {}
for idx, (class_name, count) in enumerate(class_counts.items()):
    # Temel ağırlık: toplam örnek / (sınıf sayısı * sınıf örnek sayısı)
    base_weight = total_samples / (len(class_counts) * count)
    
    # AKUT SINIFI İÇİN ÖZEL AĞIRLIK ARTIRIMI
    if class_name == "Akut":
        # Akut sınıfı için ağırlığı 3 katına çıkar
        weight = base_weight * 3.0
        print(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f} (3x artırıldı)")
    else:
        weight = base_weight
        print(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f}")
    
    class_weights[idx] = weight

print(f"\n✅ Sınıf ağırlıkları hazır: {class_weights}")

# ============================================================================
# ADIM 7: Geliştirilmiş Data Augmentation (AKUT İYİLEŞTİRME)
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 7: Geliştirilmiş Data Augmentation")
print("=" * 80)

# Akut sınıfı için daha agresif augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # 15'ten 30'a çıkarıldı
    width_shift_range=0.2,  # 0.1'den 0.2'ye çıkarıldı
    height_shift_range=0.2,  # 0.1'den 0.2'ye çıkarıldı
    shear_range=0.2,  # 0.1'den 0.2'ye çıkarıldı
    zoom_range=0.2,  # 0.1'den 0.2'ye çıkarıldı
    horizontal_flip=True,
    vertical_flip=False,  # Yeni: dikey çevirme
    fill_mode='constant',
    cval=0.0,
    brightness_range=[0.8, 1.2],  # Yeni: parlaklık değişimi
    preprocessing_function=keras.applications.efficientnet.preprocess_input,
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.efficientnet.preprocess_input,
)

print("✅ Geliştirilmiş augmentation hazır")

# ============================================================================
# ADIM 8: Data Generator'ları Oluştur
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 8: Data Generator'lar Oluşturuluyor...")
print("=" * 80)

# Model yapılandırması (20 epoch için)
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),
    "num_classes": 2,  # Akut, İnme Yok
    "batch_size": 32,  # GPU için artırıldı
    "epochs": 20,  # ⚠️ 20 EPOCH
    "learning_rate": 0.0001,
    "dropout_rate": 0.5,  # 0.7'den 0.5'e düşürüldü (daha az overfitting riski)
}

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=MODEL_CONFIG["input_shape"][:2],
    batch_size=MODEL_CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=MODEL_CONFIG["input_shape"][:2],
    batch_size=MODEL_CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False,
    seed=42,
)

print(f"✅ Train sınıfları: {train_generator.class_indices}")
print(f"✅ Train örnek sayısı: {train_generator.samples}")
print(f"✅ Validation örnek sayısı: {val_generator.samples}")

# ============================================================================
# ADIM 9: Model Oluştur
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 9: Model Oluşturuluyor...")
print("=" * 80)

model = build_model(
    input_shape=MODEL_CONFIG["input_shape"],
    num_classes=MODEL_CONFIG["num_classes"],
    base_model_name="EfficientNetB3",
    dropout_rate=MODEL_CONFIG["dropout_rate"],
    trainable_layers=-1,  # Tüm katmanlar eğitilebilir
    weights="imagenet",
)

print(f"✅ Model oluşturuldu: {model.name}")
print(f"📊 Toplam parametre: {model.count_params():,}")

# F1 Score metriği
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Modeli derle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"]),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        F1Score(),
    ],
)

print("✅ Model derlendi")

# ============================================================================
# ADIM 10: Callback'leri Oluştur
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 10: Callback'ler Oluşturuluyor...")
print("=" * 80)

# Model checkpoint
checkpoint_dir = config.CHECKPOINTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_dir / "best_model_akut_20epoch.h5"),
    monitor='val_f1_score',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1,
)

# Early stopping (20 epoch için)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,  # 20 epoch için uygun
    restore_best_weights=True,
    verbose=1,
)

# Learning rate reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,  # 20 epoch için uygun
    min_lr=1e-7,
    verbose=1,
)

# TensorBoard
tensorboard_dir = config.LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=str(tensorboard_dir),
    histogram_freq=1,
    write_graph=True,
    write_images=True,
)

callbacks = [checkpoint_callback, early_stopping, reduce_lr, tensorboard_callback]

print("✅ Callback'ler hazır")

# ============================================================================
# ADIM 11: EĞİTİM BAŞLAT (20 EPOCH)
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 11: EĞİTİM BAŞLATILIYOR... 🚀")
print("=" * 80)
print(f"⚠️ 20 EPOCH - Bu işlem 1.5-3 saat sürebilir!")
print(f"📊 Batch size: {MODEL_CONFIG['batch_size']}")
print(f"📊 Learning rate: {MODEL_CONFIG['learning_rate']}")
print(f"📊 Sınıf ağırlıkları: {class_weights}")
print(f"📊 Sadece Akut ve İnme Yok sınıfları kullanılıyor")
print("=" * 80)

# Eğitim
history = model.fit(
    train_generator,
    epochs=MODEL_CONFIG["epochs"],
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

print("\n✅ Eğitim tamamlandı!")

# ============================================================================
# ADIM 12: Modeli Kaydet
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 12: Model Kaydediliyor...")
print("=" * 80)

# En iyi modeli kaydet
best_model_path = config.SAVED_MODELS_DIR / "best_model_akut_20epoch.h5"
model.save(str(best_model_path))
print(f"✅ Model kaydedildi: {best_model_path}")

# Eğitim geçmişini kaydet
history_dict = {}
for key, values in history.history.items():
    history_dict[key] = [float(v) for v in values]

history_path = config.RESULTS_DIR / "training_history_akut_20epoch.json"
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history_dict, f, indent=2, ensure_ascii=False)
print(f"✅ Eğitim geçmişi kaydedildi: {history_path}")

# ============================================================================
# ADIM 13: Test Seti Değerlendirmesi
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 13: Test Seti Değerlendirmesi")
print("=" * 80)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.efficientnet.preprocess_input,
)

test_generator = test_datagen.flow_from_directory(
    config.TEST_DIR,
    target_size=MODEL_CONFIG["input_shape"][:2],
    batch_size=MODEL_CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False,
)

evaluator = ModelEvaluator(model)
results = evaluator.evaluate(
    test_generator,
    save_results=True,
    output_dir=config.RESULTS_DIR,
)

# Sonuçları yazdır
print("\n" + "=" * 80)
print("🎉 EĞİTİM SONUÇLARI")
print("=" * 80)
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"Precision: {results['metrics']['precision']:.4f}")
print(f"Recall: {results['metrics']['recall']:.4f}")
print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
print("\n📊 Sınıf Bazında Sonuçlar:")
if 'per_class' in results['metrics']:
    for class_name, metrics in results['metrics']['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")
print("=" * 80)

print("\n🎉 TÜM İŞLEMLER TAMAMLANDI!")
print(f"📁 Model: {best_model_path}")
print(f"📁 Sonuçlar: {config.RESULTS_DIR}")
print(f"📊 Epoch: 20")
print(f"📊 Sınıflar: Akut, İnme Yok")

