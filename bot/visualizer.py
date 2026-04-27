"""
Görselleştirme ve Raporlama Modülü
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, df: pd.DataFrame, number_columns: list):
        self.df = df
        self.number_columns = number_columns
        os.makedirs('outputs/plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    
    def plot_frequency_distribution(self):
        """Sayı frekans dağılımı"""
        all_nums = self.df[self.number_columns].values.flatten()
        frequencies = Counter(all_nums)
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(1, 81), [frequencies.get(i, 0) for i in range(1, 81)])
        
        # Renklendirme: en sık 10 kırmızı, en az 10 mavi
        sorted_freq = sorted(frequencies.items(), key=lambda x: x[1])
        most_common = [num for num, _ in sorted_freq[-10:]]
        least_common = [num for num, _ in sorted_freq[:10]]
        
        for i, bar in enumerate(bars, 1):
            if i in most_common:
                bar.set_color('red')
            elif i in least_common:
                bar.set_color('blue')
            else:
                bar.set_color('gray')
        
        plt.title('On Numara - Sayı Frekans Dağılımı (1-80)', fontsize=14)
        plt.xlabel('Sayılar', fontsize=12)
        plt.ylabel('Çıkma Sayısı', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/plots/frequency_distribution.png', dpi=150)
        plt.close()
        print("📊 Frekans grafiği kaydedildi")
    
    def plot_number_trends(self):
        """Zaman içinde sayı trendleri"""
        # En sık çıkan 16 sayıyı seç
        all_nums = self.df[self.number_columns].values.flatten()
        most_common = [num for num, _ in Counter(all_nums).most_common(16)]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, num in enumerate(most_common):
            ax = axes[idx]
            appearances = []
            for i, row in self.df.iterrows():
                if num in row[self.number_columns].values:
                    appearances.append(i)
            
            # Hareketli ortalama
            if len(appearances) > 20:
                window = 20
                smooth = np.convolve([1]*window, appearances, mode='valid') / window
                ax.plot(range(window-1, len(appearances)), smooth, 'r-', alpha=0.7)
            
            ax.scatter(range(len(appearances)), appearances, s=10, alpha=0.5)
            ax.set_title(f'Sayı {num} (toplam: {len(appearances)})', fontsize=10)
            ax.set_xlabel('Çıkış sırası')
            ax.set_ylabel('Çekiliş No')
        
        plt.suptitle('En Sık Çıkan Sayıların Zaman İçindeki Dağılımı', fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/plots/number_trends.png', dpi=150)
        plt.close()
        print("📈 Trend grafiği kaydedildi")
    
    def plot_heatmap(self):
        """Sayı birliktelik ısı haritası"""
        # En sık çıkan 20 sayı için birliktelik matrisi
        all_nums = self.df[self.number_columns].values.flatten()
        top_20 = [num for num, _ in Counter(all_nums).most_common(20)]
        
        cooccurrence = np.zeros((20, 20))
        
        for _, row in self.df.iterrows():
            nums = row[self.number_columns].values
            for i, n1 in enumerate(nums):
                if n1 in top_20:
                    for n2 in nums[i+1:]:
                        if n2 in top_20:
                            idx1, idx2 = top_20.index(n1), top_20.index(n2)
                            cooccurrence[idx1, idx2] += 1
                            cooccurrence[idx2, idx1] += 1
        
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cooccurrence, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(20))
        ax.set_yticks(range(20))
        ax.set_xticklabels(top_20)
        ax.set_yticklabels(top_20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.colorbar(im, ax=ax, label='Birlikte çıkma sayısı')
        plt.title('En Sık 20 Sayının Birliktelik Isı Haritası', fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/plots/cooccurrence_heatmap.png', dpi=150)
        plt.close()
        print("🔥 Isı haritası kaydedildi")
    
    def generate_weekly_report(self, backtest_results: dict, future_predictions: list):
        """Haftalık rapor oluştur"""
        report_lines = []
        report_lines.append("# 📊 On Numara Haftalık Analiz Raporu\n")
        report_lines.append(f"\n**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        report_lines.append(f"\n**Toplam Çekiliş:** {len(self.df)}")
        report_lines.append(f"\n**İlk Çekiliş:** {self.df['tarih'].min().strftime('%d.%m.%Y')}")
        report_lines.append(f"\n**Son Çekiliş:** {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        report_lines.append("\n\n---\n")
        report_lines.append("\n## 📈 Model Performansları (Backtest)\n")
        report_lines.append("\n| Model | Ortalama Doğruluk (/10) |")
        report_lines.append("\n|-------|-------------------------|")
        
        for model in sorted(backtest_results.keys(), key=lambda x: backtest_results[x]['avg_score'], reverse=True):
            avg = backtest_results[model]['avg_score']
            report_lines.append(f"\n| {model} | {avg:.2f} |")
        
        best_model = max(backtest_results.keys(), key=lambda x: backtest_results[x]['avg_score'])
        avg_all = sum(backtest_results[m]['avg_score'] for m in backtest_results) / len(backtest_results)
        
        report_lines.append(f"\n\n**En Başarılı Model:** {best_model}")
        report_lines.append(f"\n**Ortalama Performans:** {avg_all:.2f}/10")
        
        report_lines.append("\n\n---\n")
        report_lines.append("\n## 🔮 Gelecek Tahminleri\n")
        
        for pred in future_predictions:
            report_lines.append(f"\n### {pred['tahmin_no']}. Tahmin - {pred['tarih']}\n")
            report_lines.append(f"\n**Ensemble (Önerilen 10 Sayı):**")
            report_lines.append(f"\n{', '.join(map(str, pred['ensemble_top10'][:10]))}")
            report_lines.append(f"\n\n**Frekans Bazlı:** {', '.join(map(str, pred['frequency_top10'][:5]))}...")
            report_lines.append(f"\n**Markov Bazlı:** {', '.join(map(str, pred['markov_top10'][:5]))}...")
            report_lines.append(f"\n**Monte Carlo:** {', '.join(map(str, pred['monte_carlo_top10'][:5]))}...\n")
        
        report_lines.append("\n\n---\n")
        report_lines.append("\n> ⚠️ **Uyarı:** Bu tahminler tamamen istatistiksel analizlere dayanmaktadır.")
        report_lines.append(" Şans oyunları şansa dayalıdır, kesin sonuç garantisi yoktur.\n")
        
        # Raporu yaz
        with open('reports/weekly_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        print("📝 Haftalık rapor kaydedildi")
    
    def run_all(self, backtest_results: dict, future_predictions: list):
        """Tüm görselleştirmeleri çalıştır"""
        self.plot_frequency_distribution()
        self.plot_number_trends()
        self.plot_heatmap()
        self.generate_weekly_report(backtest_results, future_predictions)
