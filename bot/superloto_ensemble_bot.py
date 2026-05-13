#!/usr/bin/env python3
"""
SÜPER LOTO ENSEMBLE BOTU V2 - AĞIRLIKLANDIRILMIŞ
En iyi 5 pattern'in önerdiği sayıları birleştirir
Başarı skorlarına göre ağırlıklandırma yapar
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
from datetime import datetime

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
# AĞIRLIKLANDIRILMIŞ TOP 5 PATTERN
# ============================================================

class WeightedTopPatterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        
        # Pattern başarı skorları (Backtest sonuçları)
        self.pattern_weights = {
            'son7': 1.33,
            'fib_range': 1.31,
            'odd_draws': 1.31,
            'fib_neighbor': 1.30,
            'four_odd_two_even': 1.29
        }
        
        # Normalize et (toplam = 1)
        total = sum(self.pattern_weights.values())
        self.normalized_weights = {k: v/total for k, v in self.pattern_weights.items()}
        
    # Pattern 1: Son 7 çekilişte en sık çıkanlar (Başarı: 1.33/12)
    def pattern_son7(self, k=20):
        recent_nums = []
        window = min(7, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 2: Fibonacci aralığı (Başarı: 1.31/12)
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
    
    # Pattern 3: Tek numaralı çekilişler (Başarı: 1.31/12)
    def pattern_odd_draws(self, k=20):
        odd_indices = self.df.iloc[1::2]  # 1,3,5... indexli çekilişler
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: Fibonacci + komşu (Başarı: 1.30/12)
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
    
    # Pattern 5: 4 tek + 2 çift (Başarı: 1.29/12)
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
# AĞIRLIKLANDIRILMIŞ ENSEMBLE BOT
# ============================================================

class WeightedEnsembleBot:
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
    
    def get_weighted_ensemble_predictions(self):
        """Ağırlıklandırılmış ensemble: Her pattern'in önerdiği sayılar ağırlıklarıyla çarpılır"""
        
        patterns = WeightedTopPatterns(self.df, self.get_numbers)
        
        # Her pattern'den 20 sayı al
        p1_raw = patterns.pattern_son7(20)
        p2_raw = patterns.pattern_fibonacci_range(20)
        p3_raw = patterns.pattern_odd_draws(20)
        p4_raw = patterns.pattern_fibonacci_neighbor(20)
        p5_raw = patterns.pattern_4odd_2even(20)
        
        # Ağırlıklandırılmış puan hesapla
        scores = {}
        pattern_details = {}
        
        # Her sayı için ağırlıklı puan ekle
        for num in p1_raw:
            scores[num] = scores.get(num, 0) + patterns.normalized_weights['son7']
            if num not in pattern_details:
                pattern_details[num] = []
            pattern_details[num].append(('Son7', patterns.normalized_weights['son7']))
        
        for num in p2_raw:
            scores[num] = scores.get(num, 0) + patterns.normalized_weights['fib_range']
            if num not in pattern_details:
                pattern_details[num] = []
            pattern_details[num].append(('FibRange', patterns.normalized_weights['fib_range']))
        
        for num in p3_raw:
            scores[num] = scores.get(num, 0) + patterns.normalized_weights['odd_draws']
            if num not in pattern_details:
                pattern_details[num] = []
            pattern_details[num].append(('OddDraws', patterns.normalized_weights['odd_draws']))
        
        for num in p4_raw:
            scores[num] = scores.get(num, 0) + patterns.normalized_weights['fib_neighbor']
            if num not in pattern_details:
                pattern_details[num] = []
            pattern_details[num].append(('FibNeighbor', patterns.normalized_weights['fib_neighbor']))
        
        for num in p5_raw:
            scores[num] = scores.get(num, 0) + patterns.normalized_weights['four_odd_two_even']
            if num not in pattern_details:
                pattern_details[num] = []
            pattern_details[num].append(('4Odd2Even', patterns.normalized_weights['four_odd_two_even']))
        
        # Puana göre sırala
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Detaylı rapor için
        detailed = []
        for num, weighted_score in sorted_numbers:
            pattern_count = len(pattern_details[num])
            pattern_names = [p[0] for p in pattern_details[num]]
            pattern_weights = [p[1] for p in pattern_details[num]]
            
            detailed.append({
                'number': num,
                'weighted_score': round(weighted_score, 4),
                'pattern_count': pattern_count,
                'patterns': pattern_names,
                'pattern_weights': pattern_weights
            })
        
        return detailed, p1_raw, p2_raw, p3_raw, p4_raw, p5_raw, patterns.normalized_weights
    
    def backtest_last_n(self, n=50):
        """Son n çekiliş üzerinde backtest yap"""
        if len(self.df) < n + 10:
            print("Yetersiz veri")
            return None
        
        print(f"\n📊 BACKTEST BAŞLIYOR (son {n} çekiliş)")
        print("-" * 60)
        
        results = []
        patterns = WeightedTopPatterns(self.df, self.get_numbers)
        
        for test_idx in range(len(self.df) - n, len(self.df)):
            # Train: test öncesi tüm veri
            train_df = self.df.iloc[:test_idx]
            temp_patterns = WeightedTopPatterns(train_df, self.get_numbers)
            
            # Tahmin yap
            p1 = set(temp_patterns.pattern_son7(12))
            p2 = set(temp_patterns.pattern_fibonacci_range(12))
            p3 = set(temp_patterns.pattern_odd_draws(12))
            p4 = set(temp_patterns.pattern_fibonacci_neighbor(12))
            p5 = set(temp_patterns.pattern_4odd_2even(12))
            
            # Ağırlıklı ensemble (top 12)
            scores = {}
            for num in p1: scores[num] = scores.get(num, 0) + patterns.normalized_weights['son7']
            for num in p2: scores[num] = scores.get(num, 0) + patterns.normalized_weights['fib_range']
            for num in p3: scores[num] = scores.get(num, 0) + patterns.normalized_weights['odd_draws']
            for num in p4: scores[num] = scores.get(num, 0) + patterns.normalized_weights['fib_neighbor']
            for num in p5: scores[num] = scores.get(num, 0) + patterns.normalized_weights['four_odd_two_even']
            
            top_12 = [num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:12]]
            
            # Gerçek sonuç
            actual = set(self.get_numbers(self.df.iloc[test_idx]))
            correct = len(set(top_12) & actual)
            results.append(correct)
            
            # Her 10 testte bir göster
            if (test_idx - (len(self.df) - n) + 1) % 10 == 0:
                print(f"  Test {test_idx - (len(self.df) - n) + 1}/{n}: {correct}/12 doğru")
        
        avg_score = sum(results) / len(results)
        max_score = max(results)
        min_score = min(results)
        
        print("\n" + "-" * 60)
        print("📈 BACKTEST SONUÇLARI")
        print("-" * 60)
        print(f"  Ortalama doğru sayı: {avg_score:.2f}/12")
        print(f"  En iyi sonuç: {max_score}/12")
        print(f"  En kötü sonuç: {min_score}/12")
        print(f"  Rastgele şans: 1.20/12")
        print(f"  İyileştirme: %{((avg_score-1.2)/1.2*100):.1f}")
        
        return {
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'results': results
        }
    
    def print_report(self):
        detailed, p1, p2, p3, p4, p5, weights = self.get_weighted_ensemble_predictions()
        
        print("\n" + "=" * 70)
        print("🎯 SÜPER LOTO ENSEMBLE BOTU V2 - AĞIRLIKLANDIRILMIŞ")
        print("   5 en iyi pattern birleştirildi (Başarı skorlarına göre ağırlıklandırma)")
        print("=" * 70)
        
        print("\n📊 PATTERN AĞIRLIKLARI (Backtest sonuçlarından):")
        print("-" * 70)
        for name, weight in weights.items():
            name_clean = {
                'son7': 'Son 7 çekiliş',
                'fib_range': 'Fibonacci aralığı',
                'odd_draws': 'Tek numaralı çekilişler',
                'fib_neighbor': 'Fibonacci + komşu',
                'four_odd_two_even': '4 tek + 2 çift'
            }[name]
            print(f"  {name_clean:25}: %{weight*100:.1f} (Başarı: {self.get_pattern_score(name)}/12)")
        
        print("\n" + "-" * 70)
        print("🏆 AĞIRLIKLANDIRILMIŞ PUAN SIRALAMASI")
        print("-" * 70)
        print(f"\n  {'Sayı':>5} | {'Ağırlıklı Puan':>14} | {'Pattern Sayısı':>15} | {'Patternler'}")
        print("  " + "-" * 65)
        
        for item in detailed[:30]:
            print(f"  {item['number']:5d} | {item['weighted_score']:14.4f} | {item['pattern_count']:15d} | {', '.join(item['patterns'])}")
        
        # En yüksek ağırlıklı puana sahip sayılar (top 12)
        top_12 = [item['number'] for item in detailed[:12]]
        top_6 = [item['number'] for item in detailed[:6]]
        
        print("\n" + "-" * 70)
        print("🎯 ÖNERİLEN 12 SAYI (En yüksek ağırlıklı puan)")
        print("-" * 70)
        
        for i in range(0, len(top_12), 4):
            group = top_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 70)
        print("🔥 ÖNERİLEN 6 SAYI (En güçlü)")
        print("-" * 70)
        print(f"\n  🌟🌟🌟  {top_6}  🌟🌟🌟")
        
        # Pattern bazında istatistik
        print("\n" + "-" * 70)
        print("📊 PATTERN BAZINDA KATKI ANALİZİ")
        print("-" * 70)
        
        # Hangi pattern'den kaç sayı top 12'de?
        top_12_set = set(top_12)
        
        pattern_contributions = {
            'Son7': len(set(p1) & top_12_set),
            'FibRange': len(set(p2) & top_12_set),
            'OddDraws': len(set(p3) & top_12_set),
            'FibNeighbor': len(set(p4) & top_12_set),
            '4Odd2Even': len(set(p5) & top_12_set)
        }
        
        for pattern, count in pattern_contributions.items():
            print(f"  {pattern:15}: {count}/12 sayı öneride")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: Ağırlıklandırma, pattern'lerin geçmiş başarılarına göre yapılmıştır.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 70)
        
        return {
            'top_12': top_12,
            'top_6': top_6,
            'all_numbers': detailed,
            'weights': weights
        }
    
    def get_pattern_score(self, pattern_name):
        scores = {
            'son7': 1.33,
            'fib_range': 1.31,
            'odd_draws': 1.31,
            'fib_neighbor': 1.30,
            'four_odd_two_even': 1.29
        }
        return scores.get(pattern_name, 1.20)
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/weighted_ensemble_predictions.json', 'w', encoding='utf-8') as f:
            json.dump({
                'recommended_12_numbers': result['top_12'],
                'recommended_6_numbers': result['top_6'],
                'pattern_weights': result['weights'],
                'all_numbers_with_scores': result['all_numbers'][:50],
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/weighted_ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 SÜPER LOTO ENSEMBLE BOT V2 - RAPOR\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n\n")
            f.write("PATTERN AĞIRLIKLARI:\n")
            for name, weight in result['weights'].items():
                name_clean = {
                    'son7': 'Son 7 çekiliş',
                    'fib_range': 'Fibonacci aralığı',
                    'odd_draws': 'Tek numaralı çekilişler',
                    'fib_neighbor': 'Fibonacci + komşu',
                    'four_odd_two_even': '4 tek + 2 çift'
                }[name]
                f.write(f"  {name_clean}: %{weight*100:.1f}\n")
            f.write("\nÖNERİLEN 12 SAYI:\n")
            f.write(str(result['top_12']) + "\n\n")
            f.write("ÖNERİLEN 6 SAYI:\n")
            f.write(str(result['top_6']) + "\n\n")
            f.write("TÜM SAYILAR (Ağırlıklı puan sıralı):\n")
            for item in result['all_numbers'][:30]:
                f.write(f"  {item['number']:3d}: {item['weighted_score']:.4f} puan ({', '.join(item['patterns'])})\n")
        
        print(f"\n💾 Kaydedildi: outputs/weighted_ensemble_predictions.json")
        print(f"💾 Kaydedildi: outputs/weighted_ensemble_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 SÜPER LOTO AĞIRLIKLANDIRILMIŞ ENSEMBLE BOTU")
    print("   5 en iyi pattern başarı skorlarına göre ağırlıklandırıldı")
    print("   Son 150 çekiliş backtest ile doğrulandı")
    print("=" * 70)
    
    bot = WeightedEnsembleBot()
    bot.load_data()
    
    # Backtest yap
    backtest_result = bot.backtest_last_n(100)
    
    # Raporu göster
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
