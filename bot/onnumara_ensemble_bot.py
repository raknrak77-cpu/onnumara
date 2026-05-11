#!/usr/bin/env python3
"""
ON NUMARA ENSEMBLE BOTU
En iyi 5 pattern'in önerdiği sayıları birleştirir
Hangi sayıların çıkma olasılığı yüksek listeler
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
import warnings
warnings.filterwarnings('ignore')

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
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        elif 'Tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['Tarih'], format='%d/%m/%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        all_cols = ['tarih'] + self.number_columns
        existing_cols = [c for c in all_cols if c in self.df.columns]
        self.df = self.df.dropna(subset=existing_cols, how='any')
        self.df = self.df.reset_index(drop=True)
        
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['month'] = self.df['tarih'].dt.month
        self.df['year'] = self.df['tarih'].dt.year
        self.df['day'] = self.df['tarih'].dt.day
        
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
# EN İYİ 5 PATTERN
# ============================================================

class TopPatterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # Pattern 1: Ayın aynı günü (6.31/22)
    def pattern_month_day(self, k=25):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            # Yeterli veri yoksa hot numbers kullan
            all_nums = []
            for _, row in self.df.iterrows():
                all_nums.extend(self.get_numbers(row))
            counter = Counter(all_nums)
            return [num for num, _ in counter.most_common(k)]
        
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 2: Trend azalan sayılar (6.28/22)
    def pattern_trend_down(self, k=25, window=30):
        scores = {}
        for num in range(1, 81):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_numbers(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_numbers(row))
            
            if older_count > 0:
                trend = older_count - recent_count
            else:
                trend = 0
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # Pattern 3: Çift numaralı çekilişler (6.18/22)
    def pattern_even_draws(self, k=25):
        even_indices = self.df.iloc[::2]
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: Due numbers (en uzun süredir çıkmayan) (6.15/22)
    def pattern_due(self, k=25):
        last_seen = {num: 0 for num in range(1, 81)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    # Pattern 5: Son 30 çekiliş (6.13/22)
    def pattern_last30(self, k=25):
        recent_nums = []
        window = min(30, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# ENSEMBLE BOT
# ============================================================

class EnsembleBot:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def get_ensemble_predictions(self):
        """En iyi 5 pattern'in önerdiği tüm sayıları birleştir"""
        
        patterns = TopPatterns(self.df, self.get_numbers)
        
        # Her pattern'den 25 sayı al (toplam 125)
        p1 = patterns.pattern_month_day(25)
        p2 = patterns.pattern_trend_down(25)
        p3 = patterns.pattern_even_draws(25)
        p4 = patterns.pattern_due(25)
        p5 = patterns.pattern_last30(25)
        
        # Tüm sayıları birleştir ve frekanslarını hesapla
        all_numbers = p1 + p2 + p3 + p4 + p5
        counter = Counter(all_numbers)
        
        # Frekansa göre sırala
        sorted_numbers = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # Her sayının hangi pattern'lerde olduğunu bul
        detailed = []
        for num, freq in sorted_numbers:
            patterns_list = []
            if num in p1: patterns_list.append('AyınGünü')
            if num in p2: patterns_list.append('TrendAzalan')
            if num in p3: patterns_list.append('ÇiftÇek')
            if num in p4: patterns_list.append('Due')
            if num in p5: patterns_list.append('Son30')
            
            detailed.append({
                'number': num,
                'frequency': freq,
                'patterns': patterns_list,
                'pattern_count': len(patterns_list)
            })
        
        return detailed, p1, p2, p3, p4, p5
    
    def print_report(self):
        detailed, p1, p2, p3, p4, p5 = self.get_ensemble_predictions()
        
        print("\n" + "=" * 70)
        print("🎯 ON NUMARA ENSEMBLE BOTU")
        print("   En iyi 5 pattern birleştirildi")
        print("   Toplam 5×25 = 125 sayı havuzu (80'den fazla, tekrarlar var)")
        print("=" * 70)
        
        print("\n📊 5 PATTERN'İN ÖNERDİĞİ SAYILAR (İLK 15):")
        print("-" * 70)
        print(f"  Pattern 1 (Ayın aynı günü)     : {p1[:15]}...")
        print(f"  Pattern 2 (Trend azalan)       : {p2[:15]}...")
        print(f"  Pattern 3 (Çift çekilişler)    : {p3[:15]}...")
        print(f"  Pattern 4 (Due numbers)        : {p4[:15]}...")
        print(f"  Pattern 5 (Son 30 çekiliş)     : {p5[:15]}...")
        
        print("\n" + "-" * 70)
        print("🏆 TÜM SAYILAR (Frekans sıralı)")
        print("-" * 70)
        print(f"\n  {'Sayı':>5} | {'Görülme':>8} | {'Pattern Sayısı':>15} | {'Patternler'}")
        print("  " + "-" * 65)
        
        for item in detailed[:30]:
            print(f"  {item['number']:5d} | {item['frequency']:8d} | {item['pattern_count']:15d} | {', '.join(item['patterns'])}")
        
        # En az 3 pattern'de ortak olan sayılar
        common_3 = [item for item in detailed if item['pattern_count'] >= 3]
        common_4 = [item for item in detailed if item['pattern_count'] >= 4]
        common_5 = [item for item in detailed if item['pattern_count'] >= 5]
        
        print("\n" + "-" * 70)
        print("🎯 EN GÜÇLÜ ADAYLAR")
        print("-" * 70)
        
        if common_5:
            print(f"\n  🔥 5 pattern'de ortak ({len(common_5)} sayı): {[c['number'] for c in common_5]}")
        
        if common_4:
            print(f"\n  ⭐ 4 pattern'de ortak ({len(common_4)} sayı): {[c['number'] for c in common_4]}")
        
        if common_3:
            print(f"\n  📌 3 pattern'de ortak ({len(common_3)} sayı): {[c['number'] for c in common_3]}")
        
        # En çok görülen 22 sayı (tahmin)
        top_22 = [item['number'] for item in detailed[:22]]
        
        print("\n" + "-" * 70)
        print("🎯 ÖNERİLEN 22 SAYI (En yüksek frekans)")
        print("-" * 70)
        
        for i in range(0, len(top_22), 11):
            group = top_22[i:i+11]
            print(f"  {i+1:2d}-{i+11:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        # En çok görülen 11 sayı (yarı tahmin)
        top_11 = [item['number'] for item in detailed[:11]]
        
        print("\n" + "-" * 70)
        print("🔥 ÖNERİLEN 11 SAYI (En güçlü yarı)")
        print("-" * 70)
        print(f"\n  🌟🌟🌟  {top_11}  🌟🌟🌟")
        
        # İSTATİSTİK
        print("\n" + "-" * 70)
        print("📊 İSTATİSTİK")
        print("-" * 70)
        
        total_unique = len(detailed)
        print(f"  Toplam farklı sayı: {total_unique}/80")
        print(f"  Ortalama her sayı: {sum(c['frequency'] for c in detailed)/total_unique:.2f} pattern'de")
        print(f"  Maksimum pattern: {max(c['pattern_count'] for c in detailed)}")
        
        # Pattern başarı özeti
        print("\n  Pattern Başarıları (Backtest sonuçları):")
        print(f"    Ayın aynı günü        : 6.31/22 (%+4.3)")
        print(f"    Trend azalan         : 6.28/22 (%+3.8)")
        print(f"    Çift çekilişler      : 6.18/22 (%+2.1)")
        print(f"    Due numbers          : 6.15/22 (%+1.7)")
        print(f"    Son 30 çekiliş       : 6.13/22 (%+1.3)")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: 5 pattern'in ortak önerdiği sayılar listelenmiştir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 70)
        
        return {
            'top_22': top_22,
            'top_11': top_11,
            'all_numbers': detailed,
            'common_3': [c['number'] for c in common_3] if common_3 else [],
            'common_4': [c['number'] for c in common_4] if common_4 else [],
            'common_5': [c['number'] for c in common_5] if common_5 else []
        }
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/onnumara_ensemble_predictions.json', 'w', encoding='utf-8') as f:
            json.dump({
                'recommended_22_numbers': result['top_22'],
                'recommended_11_numbers': result['top_11'],
                'all_numbers_with_frequency': result['all_numbers'][:50],
                'common_in_5_patterns': result['common_5'],
                'common_in_4_patterns': result['common_4'],
                'common_in_3_patterns': result['common_3']
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/onnumara_ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 ON NUMARA ENSEMBLE BOT RAPORU\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n\n")
            f.write("ÖNERİLEN 22 SAYI:\n")
            f.write(str(result['top_22']) + "\n\n")
            f.write("ÖNERİLEN 11 SAYI (En güçlü):\n")
            f.write(str(result['top_11']) + "\n\n")
            f.write("5 PATTERN'DE ORTAK OLAN SAYILAR (Çok güçlü):\n")
            f.write(str(result['common_5']) + "\n\n")
            f.write("4 PATTERN'DE ORTAK OLAN SAYILAR:\n")
            f.write(str(result['common_4']) + "\n\n")
            f.write("3 PATTERN'DE ORTAK OLAN SAYILAR:\n")
            f.write(str(result['common_3']) + "\n\n")
            f.write("TÜM SAYILAR (Frekans sıralı - ilk 50):\n")
            for item in result['all_numbers'][:50]:
                f.write(f"  {item['number']:3d}: {item['frequency']} pattern'de ({', '.join(item['patterns'])})\n")
        
        print(f"\n💾 Kaydedildi: outputs/onnumara_ensemble_predictions.json")
        print(f"💾 Kaydedildi: outputs/onnumara_ensemble_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 ON NUMARA ENSEMBLE BOTU")
    print("   En iyi 5 pattern birleştirildi")
    print("   Toplam 5×25 = 125 sayı havuzu")
    print("=" * 70)
    
    bot = EnsembleBot()
    bot.load_data()
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
