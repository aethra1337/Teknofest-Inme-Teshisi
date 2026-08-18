"""
Model eğitim modülü
"""

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Dict, Optional, Callable
import logging
import json
from datetime import datetime

from models.stroke_classifier import StrokeClassifier
from models.model_builder import build_model
import config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model eğitim sınıfı"""
    
    def __init__(
        self,
        model_config: Dict,
        train_dir: Path,
        val_dir: Path,
        output_dir: Path,
    ):
        """
        Args:
            model_config: Model yapılandırması
            train_dir: Eğitim veri dizini
            val_dir: Validasyon veri dizini
            output_dir: Çıktı dizini
        """
        self.model_config = model_config
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        
        self.model = None
        self.history = None
    
    def create_data_generators(
        self,
        batch_size: int = 32,
        target_size: tuple = (224, 224),
        augmentation: bool = True,
    ) -> tuple:
        """
        Veri generator'ları oluşturur
        
        Args:
            batch_size: Batch boyutu
            target_size: Hedef görüntü boyutu
            augmentation: Data augmentation uygulanacak mı
            
        Returns:
            (train_generator, val_generator)
        """
        # Data augmentation parametreleri
        if augmentation:
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=config.IMAGE_PREPROCESSING["augmentation"]["rotation_range"],
                width_shift_range=config.IMAGE_PREPROCESSING["augmentation"]["width_shift_range"],
                height_shift_range=config.IMAGE_PREPROCESSING["augmentation"]["height_shift_range"],
                shear_range=config.IMAGE_PREPROCESSING["augmentation"]["shear_range"],
                zoom_range=config.IMAGE_PREPROCESSING["augmentation"]["zoom_range"],
                horizontal_flip=config.IMAGE_PREPROCESSING["augmentation"]["horizontal_flip"],
                fill_mode=config.IMAGE_PREPROCESSING["augmentation"]["fill_mode"],
                preprocessing_function=self._get_preprocessing_function(),
            )
        else:
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self._get_preprocessing_function(),
            )
        
        # Validasyon için augmentation yok
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self._get_preprocessing_function(),
        )
        
        # Generator'ları oluştur
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=config.DATA_SPLIT["random_seed"],
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=config.DATA_SPLIT["random_seed"],
        )
        
        logger.info(f"Train sınıfları: {train_generator.class_indices}")
        logger.info(f"Train örnek sayısı: {train_generator.samples}")
        logger.info(f"Validation örnek sayısı: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def _get_preprocessing_function(self) -> Optional[Callable]:
        """
        Preprocessing fonksiyonu döndürür
        
        Returns:
            Preprocessing fonksiyonu veya None
        """
        if config.IMAGE_PREPROCESSING["normalization"] == "imagenet":
            return keras.applications.efficientnet.preprocess_input
        elif config.IMAGE_PREPROCESSING["normalization"] == "custom":
            # Özel normalizasyon
            def custom_preprocess(x):
                x = x / 255.0
                mean = tf.constant([0.5, 0.5, 0.5])
                std = tf.constant([0.5, 0.5, 0.5])
                return (x - mean) / std
            return custom_preprocess
        else:
            return None
    
    def create_callbacks(self) -> list:
        """
        Eğitim callback'lerini oluşturur
        
        Returns:
            Callback listesi
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_dir = config.CHECKPOINTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.h5"),
            monitor='val_f1_score',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.MODEL_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.MODEL_CONFIG["reduce_lr_factor"],
            patience=config.MODEL_CONFIG["reduce_lr_patience"],
            min_lr=config.MODEL_CONFIG["min_lr"],
            verbose=1,
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_dir = config.LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(
        self,
        class_weights: Optional[Dict[int, float]] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
    ):
        """
        Modeli eğitir
        
        Args:
            class_weights: Sınıf ağırlıkları
            steps_per_epoch: Epoch başına adım sayısı
            validation_steps: Validasyon adım sayısı
        """
        # Model oluştur
        self.model = build_model(**self.model_config)
        
        # F1 Score metriği oluştur
        f1_metric = self._create_f1_metric()
        
        # Modeli derle
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=config.MODEL_CONFIG["learning_rate"]
            ),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                f1_metric,
            ],
        )
        
        # Data generator'ları oluştur
        train_gen, val_gen = self.create_data_generators(
            batch_size=config.MODEL_CONFIG["batch_size"],
            target_size=config.MODEL_CONFIG["input_shape"][:2],
        )
        
        # Callback'leri oluştur
        callbacks = self.create_callbacks()
        
        # Eğitim
        logger.info("Eğitim başlatılıyor...")
        self.history = self.model.fit(
            train_gen,
            epochs=config.MODEL_CONFIG["epochs"],
            validation_data=val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        
        logger.info("Eğitim tamamlandı")
        
        # En iyi modeli kaydet
        best_model_path = config.SAVED_MODELS_DIR / "best_model.h5"
        self.model.save(str(best_model_path))
        logger.info(f"En iyi model kaydedildi: {best_model_path}")
        
        # Eğitim geçmişini kaydet
        self.save_training_history()
    
    def _create_f1_metric(self):
        """F1 Score metriği oluşturur"""
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
        
        return F1Score()
    
    def save_training_history(self):
        """Eğitim geçmişini kaydeder"""
        if self.history is None:
            return
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Eğitim geçmişi kaydedildi: {history_path}")

