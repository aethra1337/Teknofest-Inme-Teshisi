"""
Model değerlendirme ve metrik hesaplama modülü
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

import config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model değerlendirme sınıfı"""
    
    def __init__(self, model, class_names: Optional[list] = None):
        """
        Args:
            model: Eğitilmiş Keras modeli
            class_names: Sınıf isimleri listesi
        """
        self.model = model
        self.class_names = class_names or config.CLASS_NAMES
    
    def evaluate(
        self,
        test_generator,
        save_results: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Modeli değerlendirir ve metrikleri hesaplar
        
        Args:
            test_generator: Test veri generator'ı
            save_results: Sonuçları kaydet
            output_dir: Çıktı dizini
            
        Returns:
            Metrikler dictionary'si
        """
        logger.info("Model değerlendiriliyor...")
        
        # Gerçek sınıf isimlerini generator'dan al
        # class_indices: {'Akut': 0, 'İnme Yok': 1} gibi bir dict
        class_indices = test_generator.class_indices
        # Sınıf isimlerini indeks sırasına göre sırala
        actual_class_names = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]
        logger.info(f"Gerçek sınıf isimleri: {actual_class_names}")
        
        # Tahminler
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Metrikleri hesapla (gerçek sınıf isimleriyle)
        metrics = self._calculate_metrics(y_true, y_pred, predictions, actual_class_names)
        
        # Classification report (sadece mevcut sınıflar için)
        report = classification_report(
            y_true,
            y_pred,
            target_names=actual_class_names,
            output_dict=True,
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Sonuçları birleştir
        results = {
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions.tolist(),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }
        
        if save_results:
            output_dir = output_dir or config.RESULTS_DIR
            self._save_results(results, output_dir)
        
        logger.info(f"Değerlendirme tamamlandı. F1 Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: list = None,
    ) -> Dict:
        """
        Metrikleri hesaplar
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            y_pred_proba: Tahmin olasılıkları
            class_names: Sınıf isimleri listesi
            
        Returns:
            Metrikler dictionary'si
        """
        if class_names is None:
            class_names = self.class_names
            
        average = config.EVALUATION_METRICS["average"]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        # Sınıf bazlı metrikler (sadece mevcut sınıflar için)
        metrics["per_class"] = {}
        unique_labels = np.unique(y_true)
        for i in unique_labels:
            if i < len(class_names):
                class_name = class_names[i]
                metrics["per_class"][class_name] = {
                    "precision": precision_score(
                        y_true, y_pred, labels=[i], average='macro', zero_division=0
                    ),
                    "recall": recall_score(
                        y_true, y_pred, labels=[i], average='macro', zero_division=0
                    ),
                    "f1_score": f1_score(
                        y_true, y_pred, labels=[i], average='macro', zero_division=0
                    ),
                }
        
        return metrics
    
    def _save_results(self, results: Dict, output_dir: Path):
        """
        Sonuçları kaydeder
        
        Args:
            results: Sonuçlar dictionary'si
            output_dir: Çıktı dizini
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrikleri kaydet
        metrics_df = pd.DataFrame([results["metrics"]])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding='utf-8-sig')
        
        # Classification report'u kaydet
        report_df = pd.DataFrame(results["classification_report"]).transpose()
        report_df.to_csv(output_dir / "classification_report.csv", encoding='utf-8-sig')
        
        # Gerçek sınıf isimlerini al (eğer test_generator'dan gelmediyse)
        # Bu durumda classification_report'dan alabiliriz
        actual_class_names = list(results["classification_report"].keys())
        # 'accuracy' ve 'macro avg' gibi özel anahtarları filtrele
        actual_class_names = [name for name in actual_class_names 
                             if name not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Confusion matrix'i görselleştir ve kaydet
        self._plot_confusion_matrix(
            results["confusion_matrix"],
            output_dir / "confusion_matrix.png",
            class_names=actual_class_names,
        )
        
        # Tahminleri kaydet
        predictions_df = pd.DataFrame({
            "true_label": [actual_class_names[i] if i < len(actual_class_names) else f"Class_{i}" 
                          for i in results["y_true"]],
            "predicted_label": [actual_class_names[i] if i < len(actual_class_names) else f"Class_{i}" 
                               for i in results["y_pred"]],
            "probabilities": results["predictions"],
        })
        predictions_df.to_csv(
            output_dir / "predictions.csv",
            index=False,
            encoding='utf-8-sig'
        )
        
        logger.info(f"Sonuçlar kaydedildi: {output_dir}")
    
    def _plot_confusion_matrix(
        self,
        cm: list,
        output_path: Path,
        class_names: list = None,
    ):
        """
        Confusion matrix'i görselleştirir
        
        Args:
            cm: Confusion matrix
            output_path: Çıktı dosya yolu
            class_names: Sınıf isimleri listesi
        """
        if class_names is None:
            class_names = self.class_names
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Gerçek Etiket', fontsize=12)
        plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix kaydedildi: {output_path}")

