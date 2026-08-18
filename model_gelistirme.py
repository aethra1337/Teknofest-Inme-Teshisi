"""
Model Geliştirme Scripti
best_stroke_detection_model.h5 modelini mevcut eğitim verileri ile geliştirir
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import json
import logging

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.data_loader import DataLoader
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / 'model_gelistirme.log', encoding='utf-8')
    ],
)
logger = logging.getLogger(__name__)

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"[OK] {len(gpus)} GPU bulundu ve yapilandirildi")
    except RuntimeError as e:
        logger.warning(f"[UYARI] GPU yapilandirma hatasi: {e}")
else:
    logger.info("[BILGI] GPU bulunamadi, CPU kullanilacak")


def evaluate_current_model():
    """Mevcut modelin performansını değerlendirir"""
    logger.info("=" * 80)
    logger.info("MEVCUT MODEL PERFORMANS DEGERLENDIRMESI")
    logger.info("=" * 80)
    
    # Modeli yükle
    model_path = Path("best_stroke_detection_model.h5")
    if not model_path.exists():
        logger.error(f"[HATA] Model bulunamadi: {model_path}")
        return None
    
    logger.info(f"[YUKLEME] Model yukleniyor: {model_path}")
    try:
        model = keras.models.load_model(str(model_path))
        logger.info("[OK] Model basariyla yuklendi!")
    except Exception as e:
        logger.error(f"[HATA] Model yukleme hatasi: {e}")
        return None
    
    # Test generator oluştur (MobileNetV2 preprocessing)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.MODEL_CONFIG["input_shape"][:2],
        batch_size=config.MODEL_CONFIG["batch_size"],
        class_mode='binary',  # Binary classification için
        shuffle=False,
    )
    
    logger.info(f"[BILGI] Test seti: {test_generator.samples} goruntu")
    logger.info(f"[BILGI] Test siniflari: {test_generator.class_indices}")
    
    # Değerlendirme (binary classification için özel)
    actual_class_names = [name for name, idx in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    
    # Binary classification için tahminler
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Metrikleri hesapla
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrikler
    report = classification_report(y_true, y_pred, target_names=actual_class_names, output_dict=True, zero_division=0)
    
    results = {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class': {}
        }
    }
    
    # Per-class metrikleri ekle
    for i, class_name in enumerate(actual_class_names):
        if str(i) in report:
            results['metrics']['per_class'][class_name] = {
                'precision': report[str(i)]['precision'],
                'recall': report[str(i)]['recall'],
                'f1_score': report[str(i)]['f1-score'],
            }
    
    # Sonuçları yazdır
    logger.info("\n" + "=" * 80)
    logger.info("MEVCUT MODEL SONUCLARI")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall:    {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score:  {results['metrics']['f1_score']:.4f}")
    
    if 'per_class' in results['metrics']:
        logger.info("\nSinif Bazli Metrikler:")
        for class_name, metrics in results['metrics']['per_class'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}")
    
    logger.info("=" * 80)
    
    return results, model


def improve_model(base_model, initial_results):
    """Modeli geliştirir"""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL GELISTIRME BASLATILIYOR")
    logger.info("=" * 80)
    
    # Veri yükleme
    logger.info("\n[VERI] Veri seti yukleniyor...")
    data_loader = DataLoader(
        iskemi_dir=config.ISKEMI_DIR,
        inmeyok_dir=config.INMEYOK_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
    )
    
    # Görüntü yollarını topla
    df = data_loader.collect_image_paths()
    
    # Sadece PNG dosyalarını kullan
    df = df[df['image_path'].apply(lambda x: str(x).lower().endswith('.png'))]
    logger.info(f"[OK] Toplam {len(df)} PNG goruntu bulundu")
    
    # Sadece Akut ve İnme Yok sınıflarını kullan (binary classification)
    logger.info("\n[FILTRE] Sinif filtreleme: Sadece Akut ve Inme Yok")
    df_filtered = df[df['label'].isin(['Akut', 'İnme Yok'])]
    logger.info(f"[OK] Filtrelenmis goruntu sayisi: {len(df_filtered)}")
    logger.info(f"[BILGI] Sinif dagilimi:")
    logger.info(df_filtered['label'].value_counts().to_string())
    
    # Veri setini böl
    logger.info("\n[BOLME] Veri seti bolunuyor...")
    train_df, val_df, test_df = data_loader.split_data(
        df_filtered,
        train_ratio=config.DATA_SPLIT["train"],
        val_ratio=config.DATA_SPLIT["val"],
        test_ratio=config.DATA_SPLIT["test"],
        random_seed=config.DATA_SPLIT["random_seed"],
    )
    
    logger.info(f"[OK] Veri bolumleme:")
    logger.info(f"  Egitim: {len(train_df)} goruntu")
    logger.info(f"  Dogrulama: {len(val_df)} goruntu")
    logger.info(f"  Test: {len(test_df)} goruntu")
    
    # Sınıf dağılımını göster
    logger.info("\n[BILGI] Sinif dagilimi (Egitim):")
    logger.info(train_df['label'].value_counts().to_string())
    
    # İşlenmiş verileri organize et
    logger.info("\n[KOPYALAMA] Goruntuler kopyalaniyor...")
    data_loader.prepare_processed_data(
        train_df,
        val_df,
        test_df,
        config.PROCESSED_DATA_DIR,
        copy_images=True,
    )
    logger.info("[OK] Veri seti hazir!")
    
    # Sınıf ağırlıklarını hesapla (Akut sınıfına daha fazla ağırlık)
    logger.info("\n[AGIRLIK] Sinif agirliklari hesaplaniyor...")
    class_counts = train_df['label'].value_counts()
    total_samples = len(train_df)
    
    # Akut sınıfına 3x ağırlık ver
    akut_weight = total_samples / (2 * class_counts.get('Akut', 1))
    inmeyok_weight = total_samples / (2 * class_counts.get('İnme Yok', 1))
    
    # Akut için ekstra ağırlık
    akut_weight = akut_weight * 3.0
    
    class_weights = {
        0: akut_weight if train_df['label'].unique()[0] == 'Akut' else inmeyok_weight,
        1: inmeyok_weight if train_df['label'].unique()[0] == 'Akut' else akut_weight,
    }
    
    logger.info(f"[BILGI] Sinif agirliklari: {class_weights}")
    
    # Model yapılandırması (Geliştirilmiş)
    model_config = {
        "input_shape": (224, 224, 3),
        "num_classes": 1,  # Binary classification
        "base_model_name": "MobileNetV2",  # Mevcut model ile aynı
        "dropout_rate": 0.6,  # Biraz düşürüldü
        "trainable_layers": 20,  # Son 20 katmanı eğitilebilir yap (fine-tuning)
        "weights": "imagenet",
    }
    
    # Mevcut modeli fine-tuning ile geliştir
    logger.info("\n[MODEL] Model gelistiriliyor...")
    
    # Mevcut modeli direkt kullan ve fine-tuning yap
    model = base_model
    
    # Son katmanları eğitilebilir yap (fine-tuning)
    model.trainable = True
    trainable_count = 0
    for layer in reversed(model.layers):
        if trainable_count < 30:  # Son 30 katmanı eğitilebilir yap
            if len(layer.get_weights()) > 0:  # Ağırlığı olan katmanlar
                layer.trainable = True
                trainable_count += 1
        else:
            layer.trainable = False
    
    logger.info(f"[OK] Model olusturuldu!")
    logger.info(f"[BILGI] Toplam parametre: {model.count_params():,}")
    logger.info(f"[BILGI] Egitilebilir parametre: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Modeli derle
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Data generators oluştur (Geliştirilmiş augmentation)
    logger.info("\n[GENERATOR] Veri generator'lari olusturuluyor...")
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # Artırıldı
        width_shift_range=0.2,  # Artırıldı
        height_shift_range=0.2,  # Artırıldı
        shear_range=0.15,  # Artırıldı
        zoom_range=0.2,  # Artırıldı
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],  # Eklendi
        fill_mode='constant',
        cval=0.0,
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        shuffle=True,
        seed=42,
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        shuffle=False,
        seed=42,
    )
    
    logger.info(f"[OK] Generator'lar olusturuldu!")
    logger.info(f"[BILGI] Egitim: {train_generator.samples} goruntu")
    logger.info(f"[BILGI] Dogrulama: {val_generator.samples} goruntu")
    
    # Callbacks
    logger.info("\n[CALLBACK] Callback'ler ayarlaniyor...")
    
    checkpoint_path = config.CHECKPOINTS_DIR / "improved_model_best.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(config.RESULTS_DIR / "improved_training_log.csv"),
            append=False,
        ),
    ]
    
    logger.info("[OK] Callback'ler ayarlandi!")
    
    # Eğitim
    logger.info("\n" + "=" * 80)
    logger.info("EGITIM BASLATILIYOR")
    logger.info("=" * 80)
    
    history = model.fit(
        train_generator,
        epochs=30,  # Daha fazla epoch
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    
    logger.info("[OK] Egitim tamamlandi!")
    
    # En iyi modeli yükle
    logger.info(f"\n[YUKLEME] En iyi model yukleniyor: {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    
    # Test değerlendirmesi
    logger.info("\n[DEGERLENDIRME] Test seti uzerinde degerlendirme yapiliyor...")
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        shuffle=False,
    )
    
    actual_class_names = [name for name, idx in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    
    # Binary classification için tahminler
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Metrikleri hesapla
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrikler
    report = classification_report(y_true, y_pred, target_names=actual_class_names, output_dict=True, zero_division=0)
    
    results = {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class': {}
        }
    }
    
    # Per-class metrikleri ekle
    for i, class_name in enumerate(actual_class_names):
        if str(i) in report:
            results['metrics']['per_class'][class_name] = {
                'precision': report[str(i)]['precision'],
                'recall': report[str(i)]['recall'],
                'f1_score': report[str(i)]['f1-score'],
            }
    
    # Sonuçları karşılaştır
    logger.info("\n" + "=" * 80)
    logger.info("SONUC KARSILASTIRMASI")
    logger.info("=" * 80)
    logger.info(f"{'Metrik':<20} {'Mevcut':<15} {'Gelistirilmis':<15} {'Fark':<15}")
    logger.info("-" * 65)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics_to_compare:
        old_val = initial_results['metrics'][metric]
        new_val = results['metrics'][metric]
        diff = new_val - old_val
        diff_pct = (diff / old_val) * 100 if old_val > 0 else 0
        logger.info(f"{metric:<20} {old_val:<15.4f} {new_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
    
    logger.info("=" * 80)
    
    # Modeli kaydet
    improved_model_path = config.SAVED_MODELS_DIR / "improved_stroke_detection_model.h5"
    model.save(str(improved_model_path))
    logger.info(f"\n[KAYIT] Gelistirilmis model kaydedildi: {improved_model_path}")
    
    # Eğitim geçmişini kaydet
    history_path = config.RESULTS_DIR / "improved_training_history.json"
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"[KAYIT] Egitim gecmisi kaydedildi: {history_path}")
    
    return results, model


def main():
    """Ana fonksiyon"""
    logger.info("=" * 80)
    logger.info("MODEL GELISTIRME SISTEMI")
    logger.info("=" * 80)
    
    # Klasörleri oluştur
    config.create_directories()
    
    # Mevcut modeli değerlendir
    initial_results, base_model = evaluate_current_model()
    
    if initial_results is None:
        logger.error("[HATA] Mevcut model degerlendirilemedi!")
        return
    
    # Modeli geliştir
    improved_results, improved_model = improve_model(base_model, initial_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("TUM ISLEMLER TAMAMLANDI!")
    logger.info("=" * 80)
    logger.info(f"[BILGI] Gelistirilmis model: {config.SAVED_MODELS_DIR / 'improved_stroke_detection_model.h5'}")
    logger.info(f"[BILGI] Sonuclar: {config.RESULTS_DIR / 'gelistirilmis_model'}")


if __name__ == "__main__":
    main()

