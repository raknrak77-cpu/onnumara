#!/usr/bin/env python3
"""
SÜPER LOTO ENSEMBLE BOTU
En iyi 5 pattern'in önerdiği sayıları birleştirir
Hangi sayıların çıkma olasılığı yüksek listeler
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os

# ============================================================
# VERİ YÜKLEYİCİ
# ============================================================

class DataLoader:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = ['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6']
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['year'] = self.df['tarih'].dt.year
        
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
    
    # Pattern 1: Son 7 çekilişte en sık çıkanlar
    def pattern_son7(self, k=20):
        recent_nums = []
        window = min(7, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 2: Fibonacci aralığı
    def pattern_fibonacci_range(self, k=20):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        fib_range = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                if x <= 60:
                    fib_range.add(x)
        # Bu sayıların geçmişteki sıklığına göre sırala
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        fib_list = list(fib_range)
        fib_list.sort(key=lambda x: counter.get(x, 0), reverse=True)
        return fib_list[:k]
    
    # Pattern 3: Tek numaralı çekilişler (1,3,5,7... çekilişler)
    def pattern_odd_draws(self, k=20):
        odd_indices = self.df.iloc[1::2]  # 1,3,5... indexli çekilişler
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: Fibonacci + komşu
    def pattern_fibonacci_neighbor(self, k=20):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 60: neighbors.add(f+1)
        neighbors = sorted(list(neighbors))
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        neighbors.sort(key=lambda x: counter.get(x, 0), reverse=True)
        return neighbors[:k]
    
    # Pattern 5: 4 tek + 2 çift (en sık 4 tek ve en sık 2 çift)
    def pattern_4odd_2even(self, k=20):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        
        top_odds = [num for num, _ in odd_counter.most_common(12)]
        top_evens = [num for num, _ in even_counter.most_common(8)]
        
        return (top_odds + top_evens)[:k]


# ============================================================
# ENSEMBLE BOT
# ============================================================

class EnsembleBot:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
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
        """5 pattern'in önerdiği tüm sayıları birleştir"""
        
        patterns = TopPatterns(self.df, self.get_numbers)
        
        # Her pattern'den 20 sayı al (toplam 100 sayı olabilir)
        p1 = patterns.pattern_son7(20)
        p2 = patterns.pattern_fibonacci_range(20)
        p3 = patterns.pattern_odd_draws(20)
        p4 = patterns.pattern_fibonacci_neighbor(20)
        p5 = patterns.pattern_4odd_2even(20)
        
        # Tüm sayıları birleştir ve frekanslarını hesapla
        all_numbers = p1 + p2 + p3 + p4 + p5
        counter = Counter(all_numbers)
        
        # Frekansa göre sırala
        sorted_numbers = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # Her sayının hangi pattern'lerde olduğunu bul
        detailed = []
        for num, freq in sorted_numbers:
            patterns_list = []
            if num in p1: patterns_list.append('Son7')
            if num in p2: patterns_list.append('FibRange')
            if num in p3: patterns_list.append('OddDraws')
            if num in p4: patterns_list.append('FibNeighbor')
            if num in p5: patterns_list.append('4Odd2Even')
            
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
        print("🎯 SÜPER LOTO ENSEMBLE BOTU")
        print("   5 en iyi pattern'in önerdiği sayılar birleştirildi")
        print("=" * 70)
        
        print("\n📊 5 PATTERN'İN ÖNERDİĞİ SAYILAR:")
        print("-" * 70)
        print(f"  Pattern 1 (Son 7 çekiliş)       : {p1[:10]}...")
        print(f"  Pattern 2 (Fibonacci aralığı)   : {p2[:10]}...")
        print(f"  Pattern 3 (Tek numaralı çek.)   : {p3[:10]}...")
        print(f"  Pattern 4 (Fibonacci + komşu)   : {p4[:10]}...")
        print(f"  Pattern 5 (4 tek + 2 çift)      : {p5[:10]}...")
        
        print("\n" + "-" * 70)
        print("🏆 TÜM SAYILAR (Frekans sıralı)")
        print("-" * 70)
        print(f"\n  {'Sayı':>5} | {'Görülme':>8} | {'Pattern Sayısı':>15} | {'Patternler'}")
        print("  " + "-" * 65)
        
        for item in detailed[:30]:
            print(f"  {item['number']:5d} | {item['frequency']:8d} | {item['pattern_count']:15d} | {', '.join(item['patterns'])}")
        
        # En az 3 pattern'de ortak olan sayılar
        common_3 = [item for item in detailed if item['pattern_count'] >= 3]
        
        print("\n" + "-" * 70)
        print("🎯 EN GÜÇLÜ ADAYLAR (En az 3 pattern'de ortak)")
        print("-" * 70)
        
        if common_3:
            print(f"\n  Toplam {len(common_3)} sayı: {[c['number'] for c in common_3]}")
        else:
            print("\n  (3 pattern'de ortak sayı bulunamadı)")
        
        # En az 2 pattern'de ortak olan sayılar
        common_2 = [item for item in detailed if item['pattern_count'] >= 2]
        
        print("\n" + "-" * 70)
        print("⭐ GÜÇLÜ ADAYLAR (En az 2 pattern'de ortak)")
        print("-" * 70)
        
        if common_2:
            two_pattern_nums = [c['number'] for c in common_2 if c['pattern_count'] == 2]
            print(f"\n  {len(common_2)} sayı: {[c['number'] for c in common_2[:20]]}")
        else:
            print("\n  (2 pattern'de ortak sayı bulunamadı)")
        
        # Tahmin: En çok görülen 12 sayı
        top_12 = [item['number'] for item in detailed[:12]]
        
        print("\n" + "-" * 70)
        print("🎯 ÖNERİLEN 12 SAYI (En yüksek frekans)")
        print("-" * 70)
        
        for i in range(0, len(top_12), 4):
            group = top_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        # Tahmin: En çok görülen 6 sayı
        top_6 = [item['number'] for item in detailed[:6]]
        
        print("\n" + "-" * 70)
        print("🔥 ÖNERİLEN 6 SAYI (En güçlü)")
        print("-" * 70)
        print(f"\n  🌟🌟🌟  {top_6}  🌟🌟🌟")
        
        # İstatistik
        print("\n" + "-" * 70)
        print("📊 İSTATİSTİK")
        print("-" * 70)
        
        total_unique = len(detailed)
        print(f"  Toplam farklı sayı: {total_unique}/60")
        print(f"  Ortalama her sayı: {sum(c['frequency'] for c in detailed)/total_unique:.2f} pattern'de")
        print(f"  Maksimum pattern: {max(c['pattern_count'] for c in detailed)}")
        
        # Pattern başarıları
        print("\n  Pattern Başarıları (Backtest):")
        print(f"    Son 7 çekiliş        : 1.33/12 (+%10.8)")
        print(f"    Fibonacci aralığı    : 1.32/12 (+%10.0)")
        print(f"    Tek numaralı çekilişler: 1.31/12 (+%9.2)")
        print(f"    Fibonacci + komşu    : 1.31/12 (+%9.2)")
        print(f"    4 tek + 2 çift       : 1.30/12 (+%8.3)")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: 5 pattern'in ortak önerdiği sayılar listelenmiştir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 70)
        
        return {
            'top_12': top_12,
            'top_6': top_6,
            'all_numbers': detailed,
            'common_3': [c['number'] for c in common_3] if common_3 else [],
            'common_2': [c['number'] for c in common_2] if common_2 else []
        }
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/ensemble_predictions.json', 'w', encoding='utf-8') as f:
            json.dump({
                'recommended_12_numbers': result['top_12'],
                'recommended_6_numbers': result['top_6'],
                'all_numbers_with_frequency': result['all_numbers'][:50],
                'common_in_3_patterns': result['common_3'],
                'common_in_2_patterns': result['common_2']
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 SÜPER LOTO ENSEMBLE BOT RAPORU\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n\n")
            f.write("ÖNERİLEN 12 SAYI:\n")
            f.write(str(result['top_12']) + "\n\n")
            f.write("ÖNERİLEN 6 SAYI:\n")
            f.write(str(result['top_6']) + "\n\n")
            f.write("EN AZ 3 PATTERN'DE ORTAK OLAN SAYILAR:\n")
            f.write(str(result['common_3']) + "\n\n")
            f.write("EN AZ 2 PATTERN'DE ORTAK OLAN SAYILAR:\n")
            f.write(str(result['common_2']) + "\n\n")
            f.write("TÜM SAYILAR (Frekans sıralı):\n")
            for item in result['all_numbers'][:30]:
                f.write(f"  {item['number']:3d}: {item['frequency']} pattern'de ({', '.join(item['patterns'])})\n")
        
        print(f"\n💾 Kaydedildi: outputs/ensemble_predictions.json")
        print(f"💾 Kaydedildi: outputs/ensemble_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 SÜPER LOTO ENSEMBLE BOTU")
    print("   5 en iyi pattern birleştirildi")
    print("   Toplam 5×20 = 100 sayı havuzu")
    print("=" * 70)
    
    bot = EnsembleBot()
    bot.load_data()
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
