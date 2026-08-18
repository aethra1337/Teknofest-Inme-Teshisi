"""
Veri seti kontrol ve analiz scripti
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.data_loader import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main():
    """Veri setini kontrol eder ve analiz eder"""
    logger.info("=" * 80)
    logger.info("Veri Seti Kontrol ve Analiz")
    logger.info("=" * 80)
    
    # Klasörleri oluştur
    config.create_directories()
    
    # Veri yükleme
    data_loader = DataLoader(
        iskemi_dir=config.ISKEMI_DIR,
        inmeyok_dir=config.INMEYOK_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
    )
    
    # Görüntü yollarını topla
    logger.info("Görüntü yolları toplanıyor...")
    df = data_loader.collect_image_paths()
    
    # Veri seti istatistikleri
    logger.info("\n" + "=" * 80)
    logger.info("VERİ SETİ İSTATİSTİKLERİ")
    logger.info("=" * 80)
    logger.info(f"Toplam görüntü sayısı: {len(df)}")
    logger.info(f"\nSınıf dağılımı:")
    logger.info(df['label'].value_counts().to_string())
    logger.info(f"\nKaynak dağılımı:")
    logger.info(df['source'].value_counts().to_string())
    
    # Veri setini böl
    logger.info("\n" + "=" * 80)
    logger.info("VERİ BÖLÜMLEME")
    logger.info("=" * 80)
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_ratio=config.DATA_SPLIT["train"],
        val_ratio=config.DATA_SPLIT["val"],
        test_ratio=config.DATA_SPLIT["test"],
        random_seed=config.DATA_SPLIT["random_seed"],
    )
    
    logger.info(f"\nTrain seti sınıf dağılımı:")
    logger.info(train_df['label'].value_counts().to_string())
    logger.info(f"\nValidation seti sınıf dağılımı:")
    logger.info(val_df['label'].value_counts().to_string())
    logger.info(f"\nTest seti sınıf dağılımı:")
    logger.info(test_df['label'].value_counts().to_string())
    
    # Sınıf ağırlıkları
    logger.info("\n" + "=" * 80)
    logger.info("SINIF AĞIRLIKLARI")
    logger.info("=" * 80)
    class_weights = data_loader.get_class_weights(train_df)
    
    logger.info("\nVeri seti kontrolü tamamlandı!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

