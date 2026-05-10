#!/usr/bin/env python3
"""
SÜPER ON NUMARA BOTU
- 6 farklı pattern (Son N, Due, Weighted, Trend, Physical, Zone)
- Ensemble mantığı ile en güçlü 22 sayı
- Backtest ile otomatik optimizasyon
- Süper Loto'daki başarılı mantığı On Numara'ya uyarlar
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
# 6 PATTERN
# ============================================================

class Patterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # Pattern 1: Son N çekiliş (N optimize edilecek)
    def pattern_recent(self, n=30, window=5):
        recent_nums = []
        window = min(window, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # Pattern 2: Due numbers (en uzun süredir çıkmayanlar)
    def pattern_due(self, n=30):
        last_seen = {num: 0 for num in range(1, 81)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # Pattern 3: Ağırlıklı frekans (yeni çekilişler daha ağırlıklı)
    def pattern_weighted(self, n=30, decay=0.95):
        weighted_counts = {}
        for idx, row in self.df.iterrows():
            weight = decay ** (len(self.df) - idx - 1)
            for num in self.get_numbers(row):
                weighted_counts[num] = weighted_counts.get(num, 0) + weight
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # Pattern 4: Trend analizi (yükselen sayılar)
    def pattern_trend(self, n=30, window=20):
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
    
    # Pattern 5: Bölgesel (1-20,21-40,41-60,61-80 dengesi)
    def pattern_zone(self, n=30):
        zones = [(1, 20), (21, 40), (41, 60), (61, 80)]
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        zone_counts = {}
        for zone in zones:
            zone_nums = [num for num in all_nums if zone[0] <= num <= zone[1]]
            zone_counts[zone] = Counter(zone_nums)
        
        all_predictions = []
        for zone in zones:
            top_from_zone = [num for num, _ in zone_counts[zone].most_common(n//4)]
            all_predictions.extend(top_from_zone)
        
        return all_predictions[:n]
    
    # Pattern 6: Son çekiliş + komşuları (Fiziksel yakınlık)
    def pattern_neighbors(self, n=30):
        last_row = self.df.iloc[-1]
        last_nums = self.get_numbers(last_row)
        neighbors = set()
        for num in last_nums:
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                neighbor = num + offset
                if 1 <= neighbor <= 80:
                    neighbors.add(neighbor)
        
        # Frekansa göre sırala
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        
        neighbor_list = list(neighbors)
        neighbor_list.sort(key=lambda x: counter.get(x, 0), reverse=True)
        
        return neighbor_list[:n]


# ============================================================
# BACKTEST ve OPTİMİZASYON
# ============================================================

class Backtest:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.results = {}
        
    def test_pattern(self, pattern_func, pattern_name, test_size=50, **kwargs):
        total = len(self.df)
        train_size = total - test_size
        
        if train_size <= 0:
            return 0
        
        scores = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            train_df = self.df.iloc[:train_end]
            temp_patterns = Patterns(train_df, self.get_numbers)
            
            if pattern_name == 'recent':
                window = kwargs.get('window', 5)
                preds = temp_patterns.pattern_recent(30, window)
            elif pattern_name == 'due':
                preds = temp_patterns.pattern_due(30)
            elif pattern_name == 'weighted':
                decay = kwargs.get('decay', 0.95)
                preds = temp_patterns.pattern_weighted(30, decay)
            elif pattern_name == 'trend':
                window = kwargs.get('window', 20)
                preds = temp_patterns.pattern_trend(30, window)
            elif pattern_name == 'zone':
                preds = temp_patterns.pattern_zone(30)
            elif pattern_name == 'neighbors':
                preds = temp_patterns.pattern_neighbors(30)
            else:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            correct = len(set(preds[:22]) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def find_best_window_for_recent(self, test_size=50):
        print("  🔍 Son N çekiliş için en iyi pencere aranıyor...")
        best_score = 0
        best_window = 5
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        
        for window in windows:
            if window > len(self.df) - test_size:
                continue
            score = self.test_pattern(None, 'recent', test_size, window=window)
            print(f"     window={window:2d} -> {score:.2f}/22 doğru")
            if score > best_score:
                best_score = score
                best_window = window
        
        print(f"  ✅ En iyi pencere: {best_window} (ortalama {best_score:.2f}/22)")
        return best_window, best_score
    
    def find_best_decay_for_weighted(self, test_size=50):
        print("  🔍 Ağırlıklı frekans için en iyi decay aranıyor...")
        best_score = 0
        best_decay = 0.95
        decays = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
        
        for decay in decays:
            score = self.test_pattern(None, 'weighted', test_size, decay=decay)
            print(f"     decay={decay} -> {score:.2f}/22 doğru")
            if score > best_score:
                best_score = score
                best_decay = decay
        
        print(f"  ✅ En iyi decay: {best_decay} (ortalama {best_score:.2f}/22)")
        return best_decay, best_score
    
    def run_all_tests(self, test_size=50):
        print(f"\n📊 BACKTEST BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("-" * 50)
        
        patterns_to_test = [
            ('recent_varsayilan', 'recent', 5),
            ('due', 'due', None),
            ('weighted_varsayilan', 'weighted', 0.95),
            ('trend', 'trend', None),
            ('zone', 'zone', None),
            ('neighbors', 'neighbors', None)
        ]
        
        print("\n  📈 PATTERN TEST SONUÇLARI:")
        for name, pattern, param in patterns_to_test:
            if param is not None:
                if name == 'recent_varsayilan':
                    score = self.test_pattern(None, pattern, test_size, window=param)
                else:
                    score = self.test_pattern(None, pattern, test_size, decay=param)
            else:
                score = self.test_pattern(None, pattern, test_size)
            self.results[name] = score
            print(f"     {name:20}: {score:.2f}/22 doğru")
        
        best_window, recent_opt_score = self.find_best_window_for_recent(test_size)
        self.results['recent_optimized'] = recent_opt_score
        self.results['recent_best_window'] = best_window
        
        best_decay, weighted_opt_score = self.find_best_decay_for_weighted(test_size)
        self.results['weighted_optimized'] = weighted_opt_score
        self.results['weighted_best_decay'] = best_decay
        
        print("\n" + "-" * 50)
        print("🏆 PATTERN BAŞARI SIRALAMASI (22'de kaç doğru):")
        print("-" * 50)
        
        sorted_results = sorted([(k, v) for k, v in self.results.items() if isinstance(v, (int, float))], 
                                key=lambda x: x[1], reverse=True)
        for i, (pattern, score) in enumerate(sorted_results[:10]):
            print(f"  {i+1}. {pattern:20}: {score:.2f}/22")
        
        return self.results


# ============================================================
# ENSEMBLE BOT (ANA TAHMİN)
# ============================================================

class SuperOnNumara:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.test_results = None
        self.optimal_settings = {}
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_backtest(self, test_size=50):
        bt = Backtest(self.df, self.get_numbers)
        self.test_results = bt.run_all_tests(test_size)
        
        self.optimal_settings = {
            'best_pattern': max([(k, v) for k, v in self.test_results.items() if isinstance(v, (int, float))], key=lambda x: x[1])[0],
            'recent_window': self.test_results.get('recent_best_window', 5),
            'weighted_decay': self.test_results.get('weighted_best_decay', 0.95)
        }
        
        print(f"\n🎯 EN İYİ PATTERN: {self.optimal_settings['best_pattern']}")
        print(f"   Son {self.optimal_settings['recent_window']} çekiliş kullanılacak (recent için)")
        return self.test_results
    
    def get_ensemble_predictions(self, n=22):
        """6 pattern'in önerilerini birleştir, en güçlü n sayıyı bul"""
        
        patterns = Patterns(self.df, self.get_numbers)
        
        recent_window = self.optimal_settings.get('recent_window', 5)
        weighted_decay = self.optimal_settings.get('weighted_decay', 0.95)
        
        # Her pattern'den 40 sayı al (kesişim için)
        p1 = patterns.pattern_recent(40, recent_window)
        p2 = patterns.pattern_due(40)
        p3 = patterns.pattern_weighted(40, weighted_decay)
        p4 = patterns.pattern_trend(40)
        p5 = patterns.pattern_zone(40)
        p6 = patterns.pattern_neighbors(40)
        
        # Tüm sayıları birleştir ve frekanslarını hesapla
        all_numbers = p1 + p2 + p3 + p4 + p5 + p6
        counter = Counter(all_numbers)
        
        # Pattern bazında ağırlıklar (backtest başarısına göre)
        weights = {
            'recent': self.test_results.get('recent_optimized', 1.0),
            'due': self.test_results.get('due', 1.0),
            'weighted': self.test_results.get('weighted_optimized', 1.0),
            'trend': self.test_results.get('trend', 1.0),
            'zone': self.test_results.get('zone', 1.0),
            'neighbors': self.test_results.get('neighbors', 1.0)
        }
        
        # Ağırlıklı skor hesapla
        weighted_scores = {}
        
        for num in range(1, 81):
            score = counter.get(num, 0)
            
            # Her pattern'deki sıralama bonusu
            for idx, (pattern_name, preds) in enumerate([('recent', p1), ('due', p2), ('weighted', p3), ('trend', p4), ('zone', p5), ('neighbors', p6)]):
                if num in preds:
                    rank = preds.index(num)
                    weight = weights.get(pattern_name, 1.0)
                    score += (40 - rank) * (weight / 10)
            
            weighted_scores[num] = score
        
        # Kesişim (en az 3 pattern'de ortak olanlar) bonusu
        intersection = set(p1) & set(p2) & set(p3) & set(p4) & set(p5) & set(p6)
        for num in intersection:
            weighted_scores[num] += 50
        
        # En az 4 pattern'de ortak olanlar
        common_4 = set()
        for num in range(1, 81):
            count = 0
            if num in p1: count += 1
            if num in p2: count += 1
            if num in p3: count += 1
            if num in p4: count += 1
            if num in p5: count += 1
            if num in p6: count += 1
            if count >= 4:
                common_4.add(num)
                weighted_scores[num] += 30
        
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        final_22 = [num for num, _ in sorted_scores[:n]]
        
        # Kesişim bilgilerini topla
        common_6 = list(intersection)
        common_5 = [num for num in range(1, 81) if sum([num in p for p in [p1,p2,p3,p4,p5,p6]]) >= 5]
        common_4_list = list(common_4)
        
        return {
            'final_22': final_22,
            'common_6': common_6,
            'common_5': common_5,
            'common_4': common_4_list,
            'pattern_scores': {k: v for k, v in self.test_results.items() if isinstance(v, (int, float))},
            'weights': weights
        }
    
    def print_report(self):
        if self.test_results is None:
            self.run_backtest()
        
        result = self.get_ensemble_predictions()
        
        print("\n" + "=" * 60)
        print("🎯 SÜPER ON NUMARA BOTU")
        print("   6 pattern + Ensemble mantığı")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🎯 En İyi Pattern: {self.optimal_settings['best_pattern']}")
        
        print("\n" + "-" * 60)
        print("🔥 KESİŞİM ANALİZİ")
        print("-" * 60)
        
        if result['common_6']:
            print(f"  6 pattern'de ortak: {result['common_6']} (SÜPER GÜÇLÜ)")
        else:
            print("  6 pattern'de ortak sayı yok")
        
        if result['common_5']:
            print(f"  5 pattern'de ortak: {result['common_5'][:10]}... ({len(result['common_5'])} adet)")
        
        if result['common_4']:
            print(f"  4 pattern'de ortak: {result['common_4'][:15]}... ({len(result['common_4'])} adet)")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 22 SAYI")
        print("-" * 60)
        
        final = result['final_22']
        for i in range(0, len(final), 5):
            group = final[i:i+5]
            print(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 ÖNERİLEN 10 SAYI (En güçlü 22'nin ilk 10'u)")
        print("-" * 60)
        print(f"\n  🌟🌟🌟  {final[:10]}  🌟🌟🌟")
        
        print("\n" + "-" * 60)
        print("📊 PATTERN BAŞARI SIRALAMASI (Backtest)")
        print("-" * 60)
        
        sorted_patterns = sorted([(k, v) for k, v in result['pattern_scores'].items() if isinstance(v, (int, float))], 
                                 key=lambda x: x[1], reverse=True)
        for i, (pattern, score) in enumerate(sorted_patterns[:8]):
            print(f"  {i+1}. {pattern:20}: {score:.2f}/22")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return result
    
    def save_results(self, result, filename="super_onnumara.json"):
        os.makedirs('outputs', exist_ok=True)
        
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump({
                'recommended_22_numbers': result['final_22'],
                'recommended_10_numbers': result['final_22'][:10],
                'common_in_6_patterns': result['common_6'],
                'common_in_5_patterns': result['common_5'],
                'common_in_4_patterns': result['common_4'],
                'pattern_performances': result['pattern_scores'],
                'optimal_settings': self.optimal_settings
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/super_onnumara_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 SÜPER ON NUMARA BOT RAPORU\n")
            f.write("🎯 SÜPER ON NUMARA BOT RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"En İyi Pattern: {self.optimal_settings['best_pattern']}\n\n")
            f.write("EN GÜÇLÜ 22 SAYI:\n")
            f.write(str(result['final_22']) + "\n\n")
            f.write("ÖNERİLEN 10 SAYI:\n")
            f.write(str(result['final_22'][:10]) + "\n\n")
            f.write("KESİŞİM ANALİZİ:\n")
            f.write(f"  6 pattern'de ortak: {result['common_6']}\n")
            f.write(f"  5 pattern'de ortak: {result['common_5'][:15]}...\n")
            f.write(f"  4 pattern'de ortak: {result['common_4'][:15]}...\n")
        
        print(f"\n💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/super_onnumara_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 SÜPER ON NUMARA BOTU")
    print("   6 pattern + Ensemble ile en güçlü 22 sayı")
    print("=" * 60)
    
    bot = SuperOnNumara()
    bot.load_data()
    bot.run_backtest(test_size=50)
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
