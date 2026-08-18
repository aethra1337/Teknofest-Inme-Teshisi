"""
TEKNOFEST 2025 - İnme Teşhisi ve Zamansal Sınıflandırma
Ana Eğitim Scripti
"""

import logging
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.data_loader import DataLoader
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
import tensorflow as tf
from tensorflow import keras

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
    """Ana eğitim fonksiyonu"""
    logger.info("=" * 80)
    logger.info("TEKNOFEST 2025 - İnme Teşhisi ve Zamansal Sınıflandırma")
    logger.info("Eğitim başlatılıyor...")
    logger.info("=" * 80)
    
    # Klasörleri oluştur
    config.create_directories()
    
    # GPU yapılandırması
    if config.GPU_CONFIG["use_gpu"]:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, config.GPU_CONFIG["gpu_memory_growth"])
                logger.info(f"{len(gpus)} GPU bulundu ve yapılandırıldı")
            except RuntimeError as e:
                logger.warning(f"GPU yapılandırma hatası: {e}")
        else:
            logger.warning("GPU bulunamadı, CPU kullanılacak")
    
    # Veri yükleme
    logger.info("Veri seti yükleniyor...")
    data_loader = DataLoader(
        iskemi_dir=config.ISKEMI_DIR,
        inmeyok_dir=config.INMEYOK_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
    )
    
    # Görüntü yollarını topla
    df = data_loader.collect_image_paths()
    
    # Veri setini böl
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_ratio=config.DATA_SPLIT["train"],
        val_ratio=config.DATA_SPLIT["val"],
        test_ratio=config.DATA_SPLIT["test"],
        random_seed=config.DATA_SPLIT["random_seed"],
    )
    
    # İşlenmiş verileri organize et (görüntüleri kopyala)
    data_loader.prepare_processed_data(
        train_df,
        val_df,
        test_df,
        config.PROCESSED_DATA_DIR,
        copy_images=True,  # Görüntüleri kopyala
    )
    
    # Sınıf ağırlıklarını hesapla
    class_weights = data_loader.get_class_weights(train_df)
    
    # Model yapılandırması
    model_config = {
        "input_shape": config.MODEL_CONFIG["input_shape"],
        "num_classes": config.MODEL_CONFIG["num_classes"],
        "base_model_name": config.TRANSFER_LEARNING["base_model"],
        "dropout_rate": config.MODEL_CONFIG["dropout_rate"],
        "trainable_layers": config.TRANSFER_LEARNING["trainable_layers"],
        "weights": config.TRANSFER_LEARNING["weights"],
    }
    
    # Eğitici oluştur
    trainer = ModelTrainer(
        model_config=model_config,
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        output_dir=config.RESULTS_DIR,
    )
    
    # Modeli eğit
    trainer.train(class_weights=class_weights)
    
    # Modeli değerlendir
    logger.info("Test seti üzerinde değerlendirme yapılıyor...")
    test_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=trainer._get_preprocessing_function(),
    ).flow_from_directory(
        config.TEST_DIR,
        target_size=config.MODEL_CONFIG["input_shape"][:2],
        batch_size=config.MODEL_CONFIG["batch_size"],
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
    logger.info("=" * 80)
    logger.info("EĞİTİM SONUÇLARI")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    logger.info("=" * 80)
    
    logger.info("Eğitim başarıyla tamamlandı!")


if __name__ == "__main__":
    main()

