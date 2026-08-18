"""
Eğitim Sonuçlarını Özetleyen Script
"""

import json
from pathlib import Path

# Eğitim geçmişini oku
history_path = Path("results/training_history.json")
if history_path.exists():
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print("=" * 80)
    print("EĞİTİM SONUÇLARI ÖZETİ")
    print("=" * 80)
    print()
    
    print("EGITIM METRIKLERI (Train Seti):")
    print("-" * 80)
    for epoch in range(len(history['loss'])):
        epoch_num = epoch + 1
        print(f"\nEpoch {epoch_num}/3:")
        print(f"  Loss:        {history['loss'][epoch]:.4f}")
        print(f"  Accuracy:    {history['accuracy'][epoch]:.4f} ({history['accuracy'][epoch]*100:.2f}%)")
        print(f"  Precision:   {history['precision'][epoch]:.4f}")
        print(f"  Recall:      {history['recall'][epoch]:.4f}")
        print(f"  F1 Score:    {history['f1_score'][epoch]:.4f}")
    
    print("\n" + "=" * 80)
    print("VALIDASYON METRIKLERI (Validation Seti):")
    print("-" * 80)
    for epoch in range(len(history['val_loss'])):
        epoch_num = epoch + 1
        print(f"\nEpoch {epoch_num}/3:")
        print(f"  Val Loss:    {history['val_loss'][epoch]:.4f}")
        print(f"  Val Accuracy: {history['val_accuracy'][epoch]:.4f} ({history['val_accuracy'][epoch]*100:.2f}%)")
        print(f"  Val Precision: {history['val_precision'][epoch]:.4f}")
        print(f"  Val Recall:  {history['val_recall'][epoch]:.4f}")
        print(f"  Val F1 Score: {history['val_f1_score'][epoch]:.4f}")
    
    print("\n" + "=" * 80)
    print("IYILESME ANALIZI:")
    print("-" * 80)
    
    # Loss iyileşmesi
    loss_improvement = history['loss'][0] - history['loss'][-1]
    print(f"\n[+] Loss Iyilesmesi: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")
    print(f"   Iyilesme: {loss_improvement:.4f} ({loss_improvement/history['loss'][0]*100:.1f}% azalma)")
    
    # Accuracy iyileşmesi
    acc_improvement = history['accuracy'][-1] - history['accuracy'][0]
    print(f"\n[+] Accuracy Iyilesmesi: {history['accuracy'][0]:.4f} -> {history['accuracy'][-1]:.4f}")
    print(f"   Iyilesme: +{acc_improvement:.4f} (+{acc_improvement*100:.2f}% artis)")
    
    # F1 Score iyileşmesi
    f1_improvement = history['f1_score'][-1] - history['f1_score'][0]
    print(f"\n[+] F1 Score Iyilesmesi: {history['f1_score'][0]:.4f} -> {history['f1_score'][-1]:.4f}")
    print(f"   Iyilesme: +{f1_improvement:.4f} (+{f1_improvement*100:.2f}% artis)")
    
    # Validation loss analizi
    print(f"\n[!] Validation Loss: {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    if history['val_loss'][-1] > history['val_loss'][0]:
        print("   [!] Validation loss artiyor - Overfitting riski var!")
    else:
        print("   [+] Validation loss azaliyor - Iyi!")
    
    print("\n" + "=" * 80)
    print("SON DURUM:")
    print("-" * 80)
    print(f"Final Train Accuracy:  {history['accuracy'][-1]:.4f} ({history['accuracy'][-1]*100:.2f}%)")
    print(f"Final Val Accuracy:    {history['val_accuracy'][-1]:.4f} ({history['val_accuracy'][-1]*100:.2f}%)")
    print(f"Final F1 Score:        {history['f1_score'][-1]:.4f}")
    print(f"Final Val F1 Score:   {history['val_f1_score'][-1]:.4f}")
    
    print("\n" + "=" * 80)
    print("DEGERLENDIRME:")
    print("-" * 80)
    
    if history['accuracy'][-1] >= 0.80:
        print("[+] Cok iyi! Model %80+ dogrulukta calisiyor")
    elif history['accuracy'][-1] >= 0.70:
        print("[+] Iyi! Model %70+ dogrulukta calisiyor")
    else:
        print("[!] Model daha fazla egitime ihtiyac duyuyor")
    
    if history['val_loss'][-1] > history['val_loss'][0]:
        print("[!] Overfitting riski: Validation loss artiyor")
        print("   Öneri: Daha fazla epoch eğitin veya dropout'u artırın")
    
    print("=" * 80)
else:
    print("Eğitim geçmişi bulunamadı!")

