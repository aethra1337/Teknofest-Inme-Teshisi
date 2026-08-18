"""
🚀 TEKNOFEST İNME TEŞHİSİ - AKUT SINIFI İYİLEŞTİRME (CPU)
Yerel Bilgisayar için CPU ile Optimize Edilmiş Eğitim
SADECE AKUT SINIFI PERFORMANSINI ARTIRMA
"""

import logging
import sys
import os
import multiprocessing
from pathlib import Path

# CPU için TensorFlow ayarları (GPU'yu devre dışı bırak)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU'yu gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Uyarıları azalt

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.data_loader import DataLoader
from models.model_builder import build_model
from evaluation.evaluator import ModelEvaluator
from training.trainer import ModelTrainer
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import json
import pandas as pd

# CPU için thread sayısını artır (hızlandırma için)
num_threads = min(multiprocessing.cpu_count(), 8)  # Maksimum 8 thread
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# Logging yapılandırması
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG["file"], encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Ana eğitim fonksiyonu - CPU ile Akut sınıfı iyileştirme"""
    logger.info("=" * 80)
    logger.info("TEKNOFEST 2025 - AKUT SINIFI İYİLEŞTİRME (CPU)")
    logger.info("CPU ile Optimize Edilmiş Eğitim")
    logger.info("=" * 80)
    
    # Klasörleri oluştur
    config.create_directories()
    
    # ============================================================================
    # CPU YAPILANDIRMASI
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("CPU YAPILANDIRMASI")
    logger.info("=" * 80)
    
    # GPU'ları kontrol et ama kullanma
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"⚠️ {len(gpus)} GPU bulundu ama CPU kullanılacak")
        logger.info("💡 CPU modu aktif (CUDA_VISIBLE_DEVICES=-1)")
    else:
        logger.info("✅ CPU modu aktif")
    
    num_threads = min(multiprocessing.cpu_count(), 8)
    logger.info(f"📊 TensorFlow versiyonu: {tf.__version__}")
    logger.info(f"📊 Thread sayısı: inter_op={num_threads}, intra_op={num_threads} (hızlandırma için artırıldı)")
    logger.info("⚠️ CPU ile eğitim GPU'dan daha yavaş olacaktır!")
    
    # ============================================================================
    # VERİ YÜKLEME VE FİLTRELEME (SADECE AKUT VE İNME YOK)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VERİ SETİ HAZIRLANIYOR (SADECE AKUT VE İNME YOK)")
    logger.info("=" * 80)
    
    data_loader = DataLoader(
        iskemi_dir=config.ISKEMI_DIR,
        inmeyok_dir=config.INMEYOK_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
    )
    
    # Görüntü yollarını topla
    logger.info("📦 Görüntü yolları toplanıyor...")
    df = data_loader.collect_image_paths()
    
    # Sadece PNG dosyalarını kullan
    df = df[df['image_path'].apply(lambda x: str(x).lower().endswith('.png'))]
    logger.info(f"✅ Toplam {len(df)} PNG görüntü bulundu")
    
    # SADECE AKUT VE İNME YOK SINIFLARINI KULLAN
    logger.info("\n🔍 Sınıf filtreleme: Sadece Akut ve İnme Yok")
    df_filtered = df[df['label'].isin(['Akut', 'İnme Yok'])]
    logger.info(f"✅ Filtrelenmiş görüntü sayısı: {len(df_filtered)}")
    logger.info(f"📊 Sınıf dağılımı:\n{df_filtered['label'].value_counts()}")
    
    # Veri setini böl
    logger.info("\n📊 Veri seti bölünüyor...")
    train_df, val_df, test_df = data_loader.split_data(
        df_filtered,  # Filtrelenmiş veri
        train_ratio=config.DATA_SPLIT["train"],
        val_ratio=config.DATA_SPLIT["val"],
        test_ratio=config.DATA_SPLIT["test"],
        random_seed=config.DATA_SPLIT["random_seed"],
    )
    
    logger.info(f"✅ Veri bölümleme:")
    logger.info(f"  Eğitim: {len(train_df)} görüntü")
    logger.info(f"  Doğrulama: {len(val_df)} görüntü")
    logger.info(f"  Test: {len(test_df)} görüntü")
    
    logger.info(f"\n📊 Sınıf dağılımı (Eğitim):\n{train_df['label'].value_counts()}")
    
    # İşlenmiş verileri organize et
    logger.info("\n📁 Görüntüler kopyalanıyor...")
    data_loader.prepare_processed_data(
        train_df,
        val_df,
        test_df,
        config.PROCESSED_DATA_DIR,
        copy_images=True,
    )
    logger.info("✅ Veri seti hazır!")
    
    # ============================================================================
    # GELİŞTİRİLMİŞ SINIF AĞIRLIKLARI (AKUT İYİLEŞTİRME - 4x ARTIRIM)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GELİŞTİRİLMİŞ SINIF AĞIRLIKLARI (AKUT İYİLEŞTİRME)")
    logger.info("=" * 80)
    
    # Sınıf sayılarını hesapla
    class_counts = train_df['label'].value_counts()
    total_samples = len(train_df)
    
    # Akut sınıfı için özel ağırlık artırımı (4x - daha agresif)
    class_weights = {}
    for idx, (class_name, count) in enumerate(class_counts.items()):
        # Temel ağırlık: toplam örnek / (sınıf sayısı * sınıf örnek sayısı)
        base_weight = total_samples / (len(class_counts) * count)
        
        # AKUT SINIFI İÇİN ÖZEL AĞIRLIK ARTIRIMI (4x)
        if class_name == "Akut":
            # Akut sınıfı için ağırlığı 4 katına çıkar (daha agresif)
            weight = base_weight * 4.0
            logger.info(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f} (4x artırıldı)")
        else:
            weight = base_weight
            logger.info(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f}")
        
        class_weights[idx] = weight
    
    logger.info(f"\n✅ Sınıf ağırlıkları hazır: {class_weights}")
    
    # ============================================================================
    # MODEL YAPILANDIRMASI (CPU İÇİN OPTİMİZE)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MODEL YAPILANDIRMASI (CPU OPTİMİZE)")
    logger.info("=" * 80)
    
    # CPU için optimize edilmiş yapılandırma (HIZLANDIRMA İÇİN)
    model_config = {
        "input_shape": (192, 192, 3),  # 224x224'ten 192x192'ye küçültüldü (hızlandırma)
        "num_classes": 2,  # Sadece Akut ve İnme Yok
        "base_model_name": "EfficientNetB0",  # B3'ten B0'a düşürüldü (daha hızlı)
        "dropout_rate": 0.6,  # Overfitting'i önlemek için
        "trainable_layers": -1,  # Tüm katmanlar eğitilebilir
        "weights": "imagenet",
    }
    
    # CPU için optimize edilmiş eğitim parametreleri (HIZLANDIRMA İÇİN)
    training_config = {
        "batch_size": 16,  # 8'den 16'ya artırıldı (daha hızlı işlem)
        "epochs": 1,  # SADECE 1 EPOCH
        "learning_rate": 0.0001,  # Düşük learning rate
        "early_stopping_patience": 20,  # Daha fazla sabır
        "reduce_lr_patience": 8,  # Learning rate azaltma sabrı
    }
    
    logger.info(f"📊 Eğitim Parametreleri (HIZLANDIRILMIŞ CPU):")
    logger.info(f"  Epochs: {training_config['epochs']}")
    logger.info(f"  Batch Size: {training_config['batch_size']} (artırıldı - daha hızlı)")
    logger.info(f"  Görüntü Boyutu: {model_config['input_shape'][:2]} (224x224'ten küçültüldü)")
    logger.info(f"  Model: {model_config['base_model_name']} (B3'ten B0'a düşürüldü - daha hızlı)")
    logger.info(f"  Learning Rate: {training_config['learning_rate']}")
    logger.info(f"  Sınıflar: Akut, İnme Yok")
    logger.info(f"  Dropout: {model_config['dropout_rate']}")
    
    # ============================================================================
    # GELİŞTİRİLMİŞ DATA AUGMENTATION (AKUT İÇİN DAHA AGRESİF)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GELİŞTİRİLMİŞ DATA AUGMENTATION (AKUT İYİLEŞTİRME)")
    logger.info("=" * 80)
    
    # Augmentation parametrelerini güncelle (daha agresif)
    config.IMAGE_PREPROCESSING["augmentation"] = {
        "rotation_range": 40,  # 30'dan 40'a çıkarıldı
        "width_shift_range": 0.25,  # 0.2'den 0.25'e çıkarıldı
        "height_shift_range": 0.25,  # 0.2'den 0.25'e çıkarıldı
        "shear_range": 0.25,  # 0.2'den 0.25'e çıkarıldı
        "zoom_range": 0.25,  # 0.2'den 0.25'e çıkarıldı
        "horizontal_flip": True,
        "vertical_flip": False,  # Tıbbi görüntüler için genelde yok
        "fill_mode": "constant",
        "cval": 0.0,
        "brightness_range": [0.7, 1.3],  # Parlaklık değişimi
    }
    
    logger.info("✅ Geliştirilmiş augmentation parametreleri ayarlandı")
    logger.info("  - Rotation: 40°")
    logger.info("  - Shift: 25%")
    logger.info("  - Shear: 25%")
    logger.info("  - Zoom: 25%")
    logger.info("  - Brightness: 0.7-1.3")
    
    # ============================================================================
    # EĞİTİCİ OLUŞTUR VE ÖZELLEŞTİR
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EĞİTİCİ OLUŞTURULUYOR")
    logger.info("=" * 80)
    
    trainer = ModelTrainer(
        model_config=model_config,
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        output_dir=config.RESULTS_DIR,
    )
    
    # Modeli oluştur
    logger.info("🔨 Model oluşturuluyor...")
    trainer.model = build_model(**model_config)
    
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
    trainer.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_config["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(),
        ],
    )
    
    logger.info(f"✅ Model oluşturuldu: {trainer.model.name}")
    logger.info(f"📊 Toplam parametre: {trainer.model.count_params():,}")
    
    # ============================================================================
    # CALLBACK'LERİ ÖZELLEŞTİR
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("CALLBACK'LER OLUŞTURULUYOR")
    logger.info("=" * 80)
    
    # Model checkpoint
    checkpoint_dir = config.CHECKPOINTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "best_model_akut_cpu.h5"),
        monitor='val_f1_score',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1,
    )
    
    # Early stopping (daha fazla sabır)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=training_config["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    
    # Learning rate reduction
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=training_config["reduce_lr_patience"],
        min_lr=1e-7,
        verbose=1,
    )
    
    # CSV Logger (ilerleme kaydı)
    csv_logger = keras.callbacks.CSVLogger(
        filename=str(config.RESULTS_DIR / "training_log_akut_cpu.csv"),
        append=False,
    )
    
    callbacks = [checkpoint_callback, early_stopping, reduce_lr, csv_logger]
    logger.info("✅ Callback'ler hazır")
    
    # ============================================================================
    # DATA GENERATOR'LARI OLUŞTUR (GELİŞTİRİLMİŞ AUGMENTATION İLE)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DATA GENERATOR'LAR OLUŞTURULUYOR")
    logger.info("=" * 80)
    
    # Geliştirilmiş augmentation ile train generator
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=config.IMAGE_PREPROCESSING["augmentation"]["rotation_range"],
        width_shift_range=config.IMAGE_PREPROCESSING["augmentation"]["width_shift_range"],
        height_shift_range=config.IMAGE_PREPROCESSING["augmentation"]["height_shift_range"],
        shear_range=config.IMAGE_PREPROCESSING["augmentation"]["shear_range"],
        zoom_range=config.IMAGE_PREPROCESSING["augmentation"]["zoom_range"],
        horizontal_flip=config.IMAGE_PREPROCESSING["augmentation"]["horizontal_flip"],
        fill_mode=config.IMAGE_PREPROCESSING["augmentation"]["fill_mode"],
        cval=config.IMAGE_PREPROCESSING["augmentation"]["cval"],
        brightness_range=config.IMAGE_PREPROCESSING["augmentation"].get("brightness_range", None),
        preprocessing_function=trainer._get_preprocessing_function(),
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=trainer._get_preprocessing_function(),
    )
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=model_config["input_shape"][:2],
        batch_size=training_config["batch_size"],
        class_mode='categorical',
        shuffle=True,
        seed=config.DATA_SPLIT["random_seed"],
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=model_config["input_shape"][:2],
        batch_size=training_config["batch_size"],
        class_mode='categorical',
        shuffle=False,
        seed=config.DATA_SPLIT["random_seed"],
    )
    
    logger.info(f"✅ Train örnek sayısı: {train_generator.samples}")
    logger.info(f"✅ Validation örnek sayısı: {val_generator.samples}")
    logger.info(f"✅ Train sınıfları: {train_generator.class_indices}")
    
    # ============================================================================
    # EĞİTİM BAŞLAT (CPU İLE)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EĞİTİM BAŞLATILIYOR... 🚀")
    logger.info("=" * 80)
    logger.info(f"⚠️ CPU MODU - SADECE 1 EPOCH (HIZLANDIRILMIŞ)")
    logger.info(f"📊 Süre tahmini: ~10-15 dakika (30 dk'dan hızlandırıldı!)")
    logger.info(f"📊 Epochs: {training_config['epochs']} (SADECE 1 EPOCH)")
    logger.info(f"📊 Batch size: {training_config['batch_size']} (artırıldı)")
    logger.info(f"📊 Görüntü boyutu: {model_config['input_shape'][:2]} (küçültüldü)")
    logger.info(f"📊 Model: {model_config['base_model_name']} (daha hızlı)")
    logger.info(f"📊 Learning rate: {training_config['learning_rate']}")
    logger.info(f"📊 Sınıf ağırlıkları: {class_weights}")
    logger.info(f"📊 Akut sınıfı: 4x ağırlıklandırıldı")
    logger.info(f"📊 Sadece Akut ve İnme Yok sınıfları kullanılıyor")
    logger.info("=" * 80)
    
    # Eğitim
    trainer.history = trainer.model.fit(
        train_generator,
        epochs=training_config["epochs"],
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    
    logger.info("\n✅ Eğitim tamamlandı!")
    
    # ============================================================================
    # MODELİ KAYDET
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MODEL KAYDEDİLİYOR")
    logger.info("=" * 80)
    
    # En iyi modeli kaydet
    best_model_path = config.SAVED_MODELS_DIR / "best_model_akut_cpu.h5"
    trainer.model.save(str(best_model_path))
    logger.info(f"✅ Model kaydedildi: {best_model_path}")
    
    # Eğitim geçmişini kaydet
    trainer.save_training_history()
    
    # Özel geçmiş dosyası
    history_path = config.RESULTS_DIR / "training_history_akut_cpu.json"
    history_dict = {}
    for key, values in trainer.history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Eğitim geçmişi kaydedildi: {history_path}")
    
    # ============================================================================
    # TEST SETİ DEĞERLENDİRMESİ
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST SETİ DEĞERLENDİRMESİ")
    logger.info("=" * 80)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=trainer._get_preprocessing_function(),
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=model_config["input_shape"][:2],
        batch_size=training_config["batch_size"],
        class_mode='categorical',
        shuffle=False,
    )
    
    # Gerçek sınıf isimleriyle değerlendirme
    actual_class_names = [name for name, idx in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    evaluator = ModelEvaluator(trainer.model, class_names=actual_class_names)
    results = evaluator.evaluate(
        test_generator,
        save_results=True,
        output_dir=config.RESULTS_DIR,
    )
    
    # Sonuçları yazdır
    logger.info("\n" + "=" * 80)
    logger.info("🎉 EĞİTİM SONUÇLARI (CPU)")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    logger.info("\n📊 Sınıf Bazında Sonuçlar:")
    if 'per_class' in results['metrics']:
        for class_name, metrics in results['metrics']['per_class'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            logger.info(f"    Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            logger.info(f"    F1 Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    logger.info("=" * 80)
    
    logger.info("\n🎉 TÜM İŞLEMLER TAMAMLANDI!")
    logger.info(f"📁 Model: {best_model_path}")
    logger.info(f"📁 Sonuçlar: {config.RESULTS_DIR}")
    logger.info(f"📊 Epoch: {training_config['epochs']} (SADECE 1 EPOCH)")
    logger.info(f"📊 Sınıflar: Akut, İnme Yok")
    logger.info(f"📊 CPU ile eğitim tamamlandı")


if __name__ == "__main__":
    main()

