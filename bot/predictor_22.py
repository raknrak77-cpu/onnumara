#!/usr/bin/env python3
"""
ON NUMARA 22'LİK SET TAHMİN BOTU
Tek dosya - Bağımsız çalışır
80 sayı içinden en güçlü 22 sayıyı önerir
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import json
import os

# ============================================================
# VERİ YÜKLEYİCİ
# ============================================================

class DataLoader:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        return self.df
    
    def get_numbers(self, row):
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums


# ============================================================
# MODELLER
# ============================================================

class Models:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # Model 1: Son 5 çekilişte en sık çıkanlar
    def recent(self, n=22):
        recent_nums = []
        window = min(5, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # Model 2: Tüm zamanların en sık çıkanları
    def frequency(self, n=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # Model 3: En uzun süredir çıkmayanlar (Due)
    def due(self, n=22):
        last_seen = {num: 0 for num in range(1, 81)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # Model 4: Ağırlıklı frekans (yeni çekilişler daha ağırlıklı)
    def weighted(self, n=22, decay=0.95):
        weighted_counts = {}
        for idx, row in self.df.iterrows():
            weight = decay ** (len(self.df) - idx - 1)
            for num in self.get_numbers(row):
                weighted_counts[num] = weighted_counts.get(num, 0) + weight
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # Model 5: Trend analizi (yükselen sayılar)
    def trend(self, n=22, window=20):
        scores = {}
        for num in range(1, 81):
            recent_count = 0
            older_count = 0
            
            recent_window = self.df.iloc[-window:]
            older_start = max(0, len(self.df) - window*2)
            older_end = len(self.df) - window
            older_window = self.df.iloc[older_start:older_end] if older_start < older_end else self.df.iloc[:window]
            
            for _, row in recent_window.iterrows():
                if num in self.get_numbers(row):
                    recent_count += 1
            for _, row in older_window.iterrows():
                if num in self.get_numbers(row):
                    older_count += 1
            
            if older_count > 0:
                trend_score = (recent_count - older_count) / older_count
            else:
                trend_score = recent_count if recent_count > 0 else 0
            
            last_seen = 0
            for idx, row in self.df.iterrows():
                if num in self.get_numbers(row):
                    last_seen = idx
            due = len(self.df) - last_seen - 1
            due_bonus = 1 / (due + 1) if due > 0 else 1
            
            scores[num] = trend_score * 10 + recent_count * 2 + due_bonus * 3
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # Model 6: Fiziksel (top ağırlığı, aşınma, kümelenme)
    def physical(self, n=22):
        scores = {}
        
        quarter = max(1, len(self.df) // 4)
        last_quarter = self.df.iloc[-quarter:]
        first_quarter = self.df.iloc[:quarter]
        
        last_counts = Counter()
        first_counts = Counter()
        
        for _, row in last_quarter.iterrows():
            last_counts.update(self.get_numbers(row))
        for _, row in first_quarter.iterrows():
            first_counts.update(self.get_numbers(row))
        
        for num in range(1, 81):
            last_freq = last_counts[num] / (quarter * 22) if quarter > 0 else 0
            first_freq = first_counts[num] / (quarter * 22) if quarter > 0 else 0
            scores[num] = (last_freq - first_freq) * 20
        
        recent_window = min(20, len(self.df))
        recent_counts = Counter()
        for _, row in self.df.iloc[-recent_window:].iterrows():
            recent_counts.update(self.get_numbers(row))
        
        for num in range(1, 81):
            recent_freq = recent_counts[num] / (recent_window * 22) if recent_counts[num] > 0 else 0.01
            scores[num] = scores.get(num, 0) + recent_freq * 10
        
        ranges = [(1, 20), (21, 40), (41, 60), (61, 80)]
        range_counts = {r: 0 for r in ranges}
        
        for _, row in self.df.iloc[-20:].iterrows():
            for num in self.get_numbers(row):
                for r in ranges:
                    if r[0] <= num <= r[1]:
                        range_counts[r] += 1
                        break
        
        best_range = max(range_counts, key=range_counts.get)
        for num in range(best_range[0], best_range[1] + 1):
            scores[num] = scores.get(num, 0) + 5
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]


# ============================================================
# ANA TAHMİN SINIFI
# ============================================================

class Predictor22:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.models = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        self.models = Models(self.df, self.get_numbers)
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def get_ensemble_22(self):
        """Tüm modelleri birleştir, en güçlü 22 sayıyı bul"""
        
        models_pred = {
            'recent': self.models.recent(30),
            'frequency': self.models.frequency(30),
            'due': self.models.due(30),
            'weighted': self.models.weighted(30),
            'trend': self.models.trend(30),
            'physical': self.models.physical(30)
        }
        
        # Kesişim
        all_sets = [set(models_pred[m]) for m in models_pred]
        intersection = list(set.intersection(*all_sets))
        
        # Ağırlıklı skor
        weighted_scores = {}
        weights = {'recent': 1.5, 'frequency': 1.0, 'due': 0.8, 'weighted': 1.2, 'trend': 1.3, 'physical': 1.4}
        
        for model_name, preds in models_pred.items():
            weight = weights.get(model_name, 1.0)
            for rank, num in enumerate(preds):
                score = (30 - rank) * weight
                weighted_scores[num] = weighted_scores.get(num, 0) + score
        
        for num in intersection:
            weighted_scores[num] = weighted_scores.get(num, 0) + 30
        
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        final_22 = [num for num, _ in sorted_scores[:22]]
        
        return {
            'final_22': final_22,
            'intersection': intersection,
            'models': models_pred,
            'scores': dict(sorted_scores[:30])
        }
    
    def print_report(self):
        result = self.get_ensemble_22()
        
        print("\n" + "=" * 60)
        print("🎯 ON NUMARA 22'LİK SET TAHMİN RAPORU")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 22 SAYI (ÖNERİLEN)")
        print("-" * 60)
        
        final = result['final_22']
        for i in range(0, len(final), 5):
            group = final[i:i+5]
            print(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 KESİŞİM (Tüm modellerin ortak önerdiği)")
        print("-" * 60)
        
        if result['intersection']:
            print(f"  {len(result['intersection'])} adet: {result['intersection']}")
        else:
            print("  (Ortak sayı bulunamadı)")
        
        print("\n" + "-" * 60)
        print("📊 MODELLERİN İLK 10 TAHMİNİ")
        print("-" * 60)
        
        for model_name, preds in result['models'].items():
            print(f"\n  {model_name.upper()}:")
            print(f"    {preds[:10]}")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return result
    
    def save_results(self, result, filename="predictions_22.json"):
        os.makedirs('outputs', exist_ok=True)
        
        # JSON kaydet
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # TXT rapor kaydet
        with open('outputs/report_22.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 ON NUMARA 22'LİK SET TAHMİN RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n\n")
            f.write("EN GÜÇLÜ 22 SAYI:\n")
            f.write(str(result['final_22']) + "\n\n")
            f.write("KESİŞİM:\n")
            f.write(str(result['intersection']) + "\n")
        
        print(f"💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/report_22.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ON NUMARA 22'LİK SET TAHMİN BOTU")
    print("   80 sayı içinden en güçlü 22 sayı")
    print("=" * 60)
    
    bot = Predictor22()
    bot.load_data()
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
