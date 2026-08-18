"""
🚀 TEKNOFEST İNME TEŞHİSİ - AKUT SINIFI İYİLEŞTİRME
Yerel Bilgisayar için GPU ile 20 Epoch Eğitim
SADECE AKUT SINIFI
"""

import logging
import sys
from pathlib import Path

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
    """Ana eğitim fonksiyonu - Sadece Akut için 20 epoch"""
    logger.info("=" * 80)
    logger.info("TEKNOFEST 2025 - AKUT SINIFI İYİLEŞTİRME (20 EPOCH)")
    logger.info("Yerel GPU ile Eğitim")
    logger.info("=" * 80)
    
    # Klasörleri oluştur
    config.create_directories()
    
    # ============================================================================
    # GPU YAPILANDIRMASI
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GPU YAPILANDIRMASI")
    logger.info("=" * 80)
    
    if config.GPU_CONFIG["use_gpu"]:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, config.GPU_CONFIG["gpu_memory_growth"])
                logger.info(f"✅ {len(gpus)} GPU bulundu ve yapılandırıldı")
                for i, gpu in enumerate(gpus):
                    logger.info(f"  GPU {i+1}: {gpu}")
            except RuntimeError as e:
                logger.warning(f"⚠️ GPU yapılandırma hatası: {e}")
        else:
            logger.warning("❌ GPU bulunamadı, CPU kullanılacak (çok yavaş olabilir!)")
            logger.warning("💡 GPU kurulumu için: python gpu_kontrol.py")
    else:
        logger.warning("⚠️ GPU kullanımı config'de kapalı!")
    
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
    # GELİŞTİRİLMİŞ SINIF AĞIRLIKLARI (AKUT İYİLEŞTİRME)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SINIF AĞIRLIKLARI HESAPLANIYOR (AKUT İYİLEŞTİRME)")
    logger.info("=" * 80)
    
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
            logger.info(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f} (3x artırıldı)")
        else:
            weight = base_weight
            logger.info(f"  {class_name}: {count} örnek → Ağırlık: {weight:.4f}")
        
        class_weights[idx] = weight
    
    logger.info(f"\n✅ Sınıf ağırlıkları hazır: {class_weights}")
    
    # ============================================================================
    # MODEL YAPILANDIRMASI (20 EPOCH)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MODEL YAPILANDIRMASI")
    logger.info("=" * 80)
    
    # 20 epoch için özel yapılandırma
    model_config = {
        "input_shape": config.MODEL_CONFIG["input_shape"],
        "num_classes": 2,  # Sadece Akut ve İnme Yok
        "base_model_name": config.TRANSFER_LEARNING["base_model"],
        "dropout_rate": 0.5,  # 0.7'den 0.5'e düşürüldü
        "trainable_layers": -1,  # Tüm katmanlar eğitilebilir
        "weights": config.TRANSFER_LEARNING["weights"],
    }
    
    # Eğitim parametreleri (20 epoch)
    training_config = {
        "batch_size": 32,  # GPU için optimize
        "epochs": 20,  # ⚠️ 20 EPOCH
        "learning_rate": 0.0001,
        "early_stopping_patience": 15,
        "reduce_lr_patience": 7,
    }
    
    logger.info(f"📊 Eğitim Parametreleri:")
    logger.info(f"  Epochs: {training_config['epochs']}")
    logger.info(f"  Batch Size: {training_config['batch_size']}")
    logger.info(f"  Learning Rate: {training_config['learning_rate']}")
    logger.info(f"  Sınıflar: Akut, İnme Yok")
    
    # ============================================================================
    # GELİŞTİRİLMİŞ DATA AUGMENTATION
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GELİŞTİRİLMİŞ DATA AUGMENTATION")
    logger.info("=" * 80)
    
    # Augmentation parametrelerini güncelle
    config.IMAGE_PREPROCESSING["augmentation"] = {
        "rotation_range": 30,  # 15'ten 30'a çıkarıldı
        "width_shift_range": 0.2,  # 0.1'den 0.2'ye çıkarıldı
        "height_shift_range": 0.2,  # 0.1'den 0.2'ye çıkarıldı
        "shear_range": 0.2,  # 0.1'den 0.2'ye çıkarıldı
        "zoom_range": 0.2,  # 0.1'den 0.2'ye çıkarıldı
        "horizontal_flip": True,
        "fill_mode": "constant",
        "cval": 0.0,
    }
    
    logger.info("✅ Geliştirilmiş augmentation parametreleri ayarlandı")
    
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
        filepath=str(checkpoint_dir / "best_model_akut_20epoch.h5"),
        monitor='val_f1_score',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1,
    )
    
    # Early stopping
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
    
    # TensorBoard
    tensorboard_dir = config.LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )
    
    callbacks = [checkpoint_callback, early_stopping, reduce_lr, tensorboard_callback]
    logger.info("✅ Callback'ler hazır")
    
    # ============================================================================
    # DATA GENERATOR'LARI OLUŞTUR
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DATA GENERATOR'LAR OLUŞTURULUYOR")
    logger.info("=" * 80)
    
    train_gen, val_gen = trainer.create_data_generators(
        batch_size=training_config["batch_size"],
        target_size=model_config["input_shape"][:2],
        augmentation=True,
    )
    
    logger.info(f"✅ Train örnek sayısı: {train_gen.samples}")
    logger.info(f"✅ Validation örnek sayısı: {val_gen.samples}")
    
    # ============================================================================
    # EĞİTİM BAŞLAT (20 EPOCH)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EĞİTİM BAŞLATILIYOR... 🚀")
    logger.info("=" * 80)
    logger.info(f"⚠️ 20 EPOCH - Bu işlem 1.5-3 saat sürebilir!")
    logger.info(f"📊 Batch size: {training_config['batch_size']}")
    logger.info(f"📊 Learning rate: {training_config['learning_rate']}")
    logger.info(f"📊 Sınıf ağırlıkları: {class_weights}")
    logger.info(f"📊 Sadece Akut ve İnme Yok sınıfları kullanılıyor")
    logger.info("=" * 80)
    
    # Eğitim
    trainer.history = trainer.model.fit(
        train_gen,
        epochs=training_config["epochs"],
        validation_data=val_gen,
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
    best_model_path = config.SAVED_MODELS_DIR / "best_model_akut_20epoch.h5"
    trainer.model.save(str(best_model_path))
    logger.info(f"✅ Model kaydedildi: {best_model_path}")
    
    # Eğitim geçmişini kaydet
    trainer.save_training_history()
    
    # Özel geçmiş dosyası
    history_path = config.RESULTS_DIR / "training_history_akut_20epoch.json"
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
    
    evaluator = ModelEvaluator(trainer.model)
    results = evaluator.evaluate(
        test_generator,
        save_results=True,
        output_dir=config.RESULTS_DIR,
    )
    
    # Sonuçları yazdır
    logger.info("\n" + "=" * 80)
    logger.info("🎉 EĞİTİM SONUÇLARI")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    logger.info("\n📊 Sınıf Bazında Sonuçlar:")
    if 'per_class' in results['metrics']:
        for class_name, metrics in results['metrics']['per_class'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
    logger.info("=" * 80)
    
    logger.info("\n🎉 TÜM İŞLEMLER TAMAMLANDI!")
    logger.info(f"📁 Model: {best_model_path}")
    logger.info(f"📁 Sonuçlar: {config.RESULTS_DIR}")
    logger.info(f"📊 Epoch: 20")
    logger.info(f"📊 Sınıflar: Akut, İnme Yok")


if __name__ == "__main__":
    main()

