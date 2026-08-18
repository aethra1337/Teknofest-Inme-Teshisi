"""
Veri Seti Analiz Scripti - PNG Dosyaları
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

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


def analyze_images(image_paths, max_samples=100):
    """Görüntüleri analiz eder"""
    logger.info(f"{len(image_paths)} görüntü analiz ediliyor (örnek: {max_samples})...")
    
    widths = []
    heights = []
    modes = []
    sizes = []
    
    sample_paths = image_paths[:max_samples] if len(image_paths) > max_samples else image_paths
    
    for img_path in sample_paths:
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes.append(str(img.mode))
                sizes.append(img_path.stat().st_size / 1024)  # KB cinsinden
        except Exception as e:
            logger.warning(f"Görüntü analiz edilemedi ({img_path}): {e}")
    
    return {
        'widths': widths,
        'heights': heights,
        'modes': modes,
        'sizes': sizes,
    }


def main():
    """Ana analiz fonksiyonu"""
    logger.info("=" * 80)
    logger.info("VERİ SETİ ANALİZİ - PNG DOSYALARI")
    logger.info("=" * 80)
    
    # Veri yükleme
    data_loader = DataLoader(
        iskemi_dir=config.ISKEMI_DIR,
        inmeyok_dir=config.INMEYOK_DIR,
        processed_dir=config.PROCESSED_DATA_DIR,
    )
    
    # Görüntü yollarını topla
    logger.info("Görüntü yolları toplanıyor...")
    df = data_loader.collect_image_paths()
    
    # Sadece PNG dosyalarını filtrele
    df['image_path_str'] = df['image_path'].astype(str)
    df = df[df['image_path_str'].str.lower().str.endswith('.png', na=False)]
    df = df.drop(columns=['image_path_str'])
    
    logger.info("\n" + "=" * 80)
    logger.info("TEMEL İSTATİSTİKLER")
    logger.info("=" * 80)
    logger.info(f"Toplam PNG görüntü sayısı: {len(df)}")
    logger.info(f"\nSınıf dağılımı:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} görüntü ({percentage:.2f}%)")
    
    logger.info(f"\nKaynak dağılımı:")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        logger.info(f"  {source}: {count} görüntü")
    
    # Görüntü analizi
    logger.info("\n" + "=" * 80)
    logger.info("GÖRÜNTÜ ANALİZİ")
    logger.info("=" * 80)
    
    image_stats = analyze_images(df['image_path'].tolist(), max_samples=500)
    
    if image_stats['widths']:
        logger.info(f"\nGörüntü boyutları (örnek: {len(image_stats['widths'])} görüntü):")
        logger.info(f"  Genişlik: min={min(image_stats['widths'])}, max={max(image_stats['widths'])}, ortalama={np.mean(image_stats['widths']):.1f}")
        logger.info(f"  Yükseklik: min={min(image_stats['heights'])}, max={max(image_stats['heights'])}, ortalama={np.mean(image_stats['heights']):.1f}")
        logger.info(f"  Dosya boyutu: min={min(image_stats['sizes']):.2f} KB, max={max(image_stats['sizes']):.2f} KB, ortalama={np.mean(image_stats['sizes']):.2f} KB")
        
        mode_counts = pd.Series(image_stats['modes']).value_counts()
        logger.info(f"\nRenk modları:")
        for mode, count in mode_counts.items():
            logger.info(f"  {mode}: {count}")
    
    # Veri bölümleme analizi
    logger.info("\n" + "=" * 80)
    logger.info("VERİ BÖLÜMLEME ANALİZİ")
    logger.info("=" * 80)
    
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_ratio=config.DATA_SPLIT["train"],
        val_ratio=config.DATA_SPLIT["val"],
        test_ratio=config.DATA_SPLIT["test"],
        random_seed=config.DATA_SPLIT["random_seed"],
    )
    
    logger.info(f"\nTrain seti: {len(train_df)} görüntü")
    train_class_counts = train_df['label'].value_counts()
    for label, count in train_class_counts.items():
        logger.info(f"  {label}: {count}")
    
    logger.info(f"\nValidation seti: {len(val_df)} görüntü")
    val_class_counts = val_df['label'].value_counts()
    for label, count in val_class_counts.items():
        logger.info(f"  {label}: {count}")
    
    logger.info(f"\nTest seti: {len(test_df)} görüntü")
    test_class_counts = test_df['label'].value_counts()
    for label, count in test_class_counts.items():
        logger.info(f"  {label}: {count}")
    
    # Sınıf dengesizliği kontrolü
    logger.info("\n" + "=" * 80)
    logger.info("SINIF DENGESİZLİĞİ ANALİZİ")
    logger.info("=" * 80)
    
    class_weights = data_loader.get_class_weights(train_df)
    logger.info("Sınıf ağırlıkları (imbalanced data için):")
    for class_idx, weight in class_weights.items():
        class_name = data_loader.reverse_class_mapping.get(class_idx, f"Sınıf {class_idx}")
        logger.info(f"  {class_name}: {weight:.4f}")
    
    # Özet
    logger.info("\n" + "=" * 80)
    logger.info("ÖZET")
    logger.info("=" * 80)
    logger.info(f"✅ Toplam {len(df)} PNG görüntü bulundu")
    logger.info(f"✅ {len(class_counts)} farklı sınıf var")
    logger.info(f"✅ Veri seti train/val/test olarak bölündü")
    logger.info(f"✅ Sınıf ağırlıkları hesaplandı")
    logger.info("\nVeri seti analizi tamamlandı!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

