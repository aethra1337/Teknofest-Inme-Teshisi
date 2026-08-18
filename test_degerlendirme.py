"""
Test Seti Değerlendirme Scripti
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import config
import tensorflow as tf
from tensorflow import keras
from evaluation.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main():
    """Test değerlendirmesi yapar"""
    logger.info("=" * 80)
    logger.info("TEST SETI DEGERLENDIRMESI")
    logger.info("=" * 80)
    
    # Modeli yükle
    model_path = config.SAVED_MODELS_DIR / "best_model.h5"
    if not model_path.exists():
        logger.error(f"Model bulunamadı: {model_path}")
        return
    
    logger.info(f"Model yükleniyor: {model_path}")
    model = keras.models.load_model(str(model_path))
    logger.info("Model yüklendi!")
    
    # Test generator oluştur
    from training.trainer import ModelTrainer
    
    # Preprocessing fonksiyonunu almak için geçici trainer
    temp_trainer = ModelTrainer(
        model_config={},
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        output_dir=config.RESULTS_DIR,
    )
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=temp_trainer._get_preprocessing_function(),
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.MODEL_CONFIG["input_shape"][:2],
        batch_size=config.MODEL_CONFIG["batch_size"],
        class_mode='categorical',
        shuffle=False,
    )
    
    logger.info(f"Test seti: {test_generator.samples} görüntü")
    logger.info(f"Test sınıfları: {test_generator.class_indices}")
    
    # Değerlendirme (gerçek sınıf isimleriyle)
    actual_class_names = [name for name, idx in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    evaluator = ModelEvaluator(model, class_names=actual_class_names)
    results = evaluator.evaluate(
        test_generator,
        save_results=True,
        output_dir=config.RESULTS_DIR,
    )
    
    # Sonuçları yazdır
    logger.info("\n" + "=" * 80)
    logger.info("TEST SONUCLARI")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall:    {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score:  {results['metrics']['f1_score']:.4f}")
    
    logger.info("\nSınıf Bazlı Metrikler:")
    for class_name, metrics in results['metrics']['per_class'].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Sonuçlar kaydedildi: results/ klasöründe")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

