"""
Eski ve Yeni Sonuçları Karşılaştırma Grafiği
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Sonuç dosyalarını oku
results_dir = Path("results")

# Eski sonuçlar (ilk eğitim)
old_metrics = {
    "Akut": {
        "precision": 0.3676,
        "recall": 0.1471,
        "f1_score": 0.2101
    },
    "İnme Yok": {
        "precision": 0.8107,
        "recall": 0.9352,
        "f1_score": 0.8685
    },
    "Genel": {
        "accuracy": 0.7746,
        "precision": 0.7204,
        "recall": 0.7746,
        "f1_score": 0.7343
    }
}

# Yeni sonuçlar (CPU eğitimi sonrası)
new_metrics = {
    "Akut": {
        "precision": 0.2903,
        "recall": 0.1059,
        "f1_score": 0.1552
    },
    "İnme Yok": {
        "precision": 0.8031,
        "recall": 0.9337,
        "f1_score": 0.8635
    },
    "Genel": {
        "accuracy": 0.7650,
        "precision": 0.6986,
        "recall": 0.7650,
        "f1_score": 0.7191
    }
}

# Grafik oluştur
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Eski vs Yeni Model Sonuçları Karşılaştırması', fontsize=16, fontweight='bold')

# 1. Genel Metrikler Karşılaştırması
ax1 = axes[0, 0]
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
old_values = [
    old_metrics["Genel"]["accuracy"],
    old_metrics["Genel"]["precision"],
    old_metrics["Genel"]["recall"],
    old_metrics["Genel"]["f1_score"]
]
new_values = [
    new_metrics["Genel"]["accuracy"],
    new_metrics["Genel"]["precision"],
    new_metrics["Genel"]["recall"],
    new_metrics["Genel"]["f1_score"]
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, old_values, width, label='Eski Model', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, new_values, width, label='Yeni Model (CPU)', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Değer', fontsize=12)
ax1.set_title('Genel Performans Metrikleri', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11)
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Değerleri çubukların üzerine yaz
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# 2. Akut Sınıfı Karşılaştırması
ax2 = axes[0, 1]
akut_metrics = ['Precision', 'Recall', 'F1 Score']
akut_old = [
    old_metrics["Akut"]["precision"],
    old_metrics["Akut"]["recall"],
    old_metrics["Akut"]["f1_score"]
]
akut_new = [
    new_metrics["Akut"]["precision"],
    new_metrics["Akut"]["recall"],
    new_metrics["Akut"]["f1_score"]
]

x2 = np.arange(len(akut_metrics))
bars3 = ax2.bar(x2 - width/2, akut_old, width, label='Eski Model', color='#3498db', alpha=0.8)
bars4 = ax2.bar(x2 + width/2, akut_new, width, label='Yeni Model (CPU)', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Değer', fontsize=12)
ax2.set_title('Akut Sınıfı Performansı', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(akut_metrics, fontsize=11)
ax2.legend(fontsize=11)
ax2.set_ylim([0, 0.5])
ax2.grid(axis='y', alpha=0.3)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# 3. İnme Yok Sınıfı Karşılaştırması
ax3 = axes[1, 0]
inmeyok_metrics = ['Precision', 'Recall', 'F1 Score']
inmeyok_old = [
    old_metrics["İnme Yok"]["precision"],
    old_metrics["İnme Yok"]["recall"],
    old_metrics["İnme Yok"]["f1_score"]
]
inmeyok_new = [
    new_metrics["İnme Yok"]["precision"],
    new_metrics["İnme Yok"]["recall"],
    new_metrics["İnme Yok"]["f1_score"]
]

x3 = np.arange(len(inmeyok_metrics))
bars5 = ax3.bar(x3 - width/2, inmeyok_old, width, label='Eski Model', color='#3498db', alpha=0.8)
bars6 = ax3.bar(x3 + width/2, inmeyok_new, width, label='Yeni Model (CPU)', color='#e74c3c', alpha=0.8)

ax3.set_ylabel('Değer', fontsize=12)
ax3.set_title('İnme Yok Sınıfı Performansı', fontsize=14, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(inmeyok_metrics, fontsize=11)
ax3.legend(fontsize=11)
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)

for bars in [bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# 4. Değişim Yüzdesi
ax4 = axes[1, 1]
changes = {
    'Genel Accuracy': ((new_metrics["Genel"]["accuracy"] - old_metrics["Genel"]["accuracy"]) / old_metrics["Genel"]["accuracy"]) * 100,
    'Genel F1': ((new_metrics["Genel"]["f1_score"] - old_metrics["Genel"]["f1_score"]) / old_metrics["Genel"]["f1_score"]) * 100,
    'Akut Recall': ((new_metrics["Akut"]["recall"] - old_metrics["Akut"]["recall"]) / old_metrics["Akut"]["recall"]) * 100,
    'Akut F1': ((new_metrics["Akut"]["f1_score"] - old_metrics["Akut"]["f1_score"]) / old_metrics["Akut"]["f1_score"]) * 100,
    'İnme Yok F1': ((new_metrics["İnme Yok"]["f1_score"] - old_metrics["İnme Yok"]["f1_score"]) / old_metrics["İnme Yok"]["f1_score"]) * 100,
}

change_labels = list(changes.keys())
change_values = list(changes.values())
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in change_values]

bars7 = ax4.barh(change_labels, change_values, color=colors, alpha=0.8)
ax4.set_xlabel('Değişim Yüzdesi (%)', fontsize=12)
ax4.set_title('Performans Değişimi', fontsize=14, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars7, change_values)):
    ax4.text(val, i, f'{val:+.2f}%',
            ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# Grafiği kaydet
output_path = results_dir / "sonuc_karsilastirma_grafigi.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Grafik kaydedildi: {output_path}")

# Ayrıca PDF olarak da kaydet
output_path_pdf = results_dir / "sonuc_karsilastirma_grafigi.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"[OK] PDF kaydedildi: {output_path_pdf}")

plt.close()

print("\n" + "="*80)
print("GRAFIK OZETI")
print("="*80)
print(f"\n[GENEL PERFORMANS]")
print(f"  Eski Model Accuracy: {old_metrics['Genel']['accuracy']:.4f} ({old_metrics['Genel']['accuracy']*100:.2f}%)")
print(f"  Yeni Model Accuracy: {new_metrics['Genel']['accuracy']:.4f} ({new_metrics['Genel']['accuracy']*100:.2f}%)")
print(f"  Degisim: {changes['Genel Accuracy']:+.2f}%")

print(f"\n[AKUT SINIFI]")
print(f"  Eski Recall: {old_metrics['Akut']['recall']:.4f} ({old_metrics['Akut']['recall']*100:.2f}%)")
print(f"  Yeni Recall: {new_metrics['Akut']['recall']:.4f} ({new_metrics['Akut']['recall']*100:.2f}%)")
print(f"  Degisim: {changes['Akut Recall']:+.2f}%")

print(f"\n[INME YOK SINIFI]")
print(f"  Eski F1: {old_metrics['İnme Yok']['f1_score']:.4f} ({old_metrics['İnme Yok']['f1_score']*100:.2f}%)")
print(f"  Yeni F1: {new_metrics['İnme Yok']['f1_score']:.4f} ({new_metrics['İnme Yok']['f1_score']*100:.2f}%)")
print(f"  Degisim: {changes['İnme Yok F1']:+.2f}%")
print("="*80)

