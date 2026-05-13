#!/usr/bin/env python3
"""
SÜPER LOTO ENSEMBLE BOT V2 - GEMİNİ OPTİMİZASYONLU
- Sadece 1.35+ başarılı pattern'ler
- 100 adet optimized kombinasyon
- Gemini önerileri: Pairwise, Overdue, Entropy, Block Gap
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import random
import math
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
# YÜKSEK BAŞARILI PATTERN'LER (1.35+)
# ============================================================

class HighPerformancePatterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        
        # SADECE 1.35+ puan alan pattern'ler (V2 backtest sonuçları)
        self.patterns_info = {
            'odd_draws': {
                'name': 'Tek numaralı çekilişler',
                'score': 1.44,
                'weight': 0.205,
                'k': 20
            },
            'low_entropy': {
                'name': 'Düşük entropi (Düzenli sayılar)',
                'score': 1.44,
                'weight': 0.205,
                'k': 20
            },
            'son7': {
                'name': 'Son 7 çekiliş',
                'score': 1.44,
                'weight': 0.205,
                'k': 20
            },
            'four_odd_two_even': {
                'name': '4 tek + 2 çift',
                'score': 1.42,
                'weight': 0.202,
                'k': 20
            },
            'son5': {
                'name': 'Son 5 çekiliş',
                'score': 1.39,
                'weight': 0.198,
                'k': 20
            },
            'overdue': {
                'name': 'Vadesi geçmiş sayılar (Gemini)',
                'score': 1.38,
                'weight': 0.197,
                'k': 20
            },
            'three_small_three_large': {
                'name': '3 küçük + 3 büyük',
                'score': 1.38,
                'weight': 0.197,
                'k': 20
            },
            'three_odd_three_even': {
                'name': '3 tek + 3 çift',
                'score': 1.37,
                'weight': 0.195,
                'k': 20
            },
            'hot': {
                'name': 'Hot numbers',
                'score': 1.36,
                'weight': 0.194,
                'k': 20
            },
            'only_odd': {
                'name': 'Sadece tek sayılar',
                'score': 1.36,
                'weight': 0.194,
                'k': 20
            },
            'son30': {
                'name': 'Son 30 çekiliş',
                'score': 1.36,
                'weight': 0.194,
                'k': 20
            },
            'trend_up': {
                'name': 'Trend artan sayılar',
                'score': 1.35,
                'weight': 0.192,
                'k': 20
            },
            # GEMİNİ ÖNERİSİ - YENİ PATTERN'LER
            'top_pairs': {
                'name': 'En çok çıkan sayı çiftleri (Gemini-Pair)',
                'score': 1.33,
                'weight': 0.189,
                'k': 20
            },
            'rhythmic_due': {
                'name': 'Ritmik Due analizi (Gemini-Rhythm)',
                'score': 1.32,
                'weight': 0.188,
                'k': 20
            },
            'high_entropy': {
                'name': 'Yüksek entropi (Gemini-Dağınık)',
                'score': 1.30,
                'weight': 0.185,
                'k': 20
            }
        }
        
        # Toplam ağırlık
        total_weight = sum(p['weight'] for p in self.patterns_info.values())
        print(f"   📊 Toplam {len(self.patterns_info)} pattern (1.30+), toplam ağırlık: {total_weight:.3f}")
    
    # Pattern 1: Tek numaralı çekilişler
    def pattern_odd_draws(self, k=20):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 2: Düşük entropi
    def pattern_low_entropy(self, k=20):
        def calculate_entropy(numbers):
            if not numbers:
                return 0
            ranges = [0] * 6
            for num in numbers:
                idx = (num - 1) // 10
                if 0 <= idx < 6:
                    ranges[idx] += 1
            entropy = 0
            total = len(numbers)
            for count in ranges:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            return entropy
        
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if calculate_entropy(nums) < 2.0:
                all_nums.extend(nums)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 3: Son 7 çekiliş
    def pattern_son7(self, k=20):
        recent_nums = []
        window = min(7, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: 4 tek + 2 çift
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
    
    # Pattern 5: Son 5 çekiliş
    def pattern_son5(self, k=20):
        recent_nums = []
        window = min(5, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 6: Overdue (Gemini)
    def pattern_overdue(self, k=20):
        last_seen = {num: 0 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        
        intervals = {num: [] for num in range(1, 61)}
        last_pos = {num: -1 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                if last_pos[num] != -1:
                    intervals[num].append(idx - last_pos[num])
                last_pos[num] = idx
        
        avg_intervals = {}
        for num in range(1, 61):
            if intervals[num]:
                avg_intervals[num] = sum(intervals[num]) / len(intervals[num])
            else:
                avg_intervals[num] = 10
        
        scores = {}
        for num in range(1, 61):
            gap = last_idx - last_seen[num]
            ratio = gap / avg_intervals[num] if avg_intervals[num] > 0 else 1
            scores[num] = ratio
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # Pattern 7: 3 küçük + 3 büyük
    def pattern_three_small_three_large(self, k=20):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        small = [num for num in all_nums if num <= 30]
        large = [num for num in all_nums if num > 30]
        
        small_counter = Counter(small)
        large_counter = Counter(large)
        
        top_small = [num for num, _ in small_counter.most_common(10)]
        top_large = [num for num, _ in large_counter.most_common(10)]
        
        return (top_small + top_large)[:k]
    
    # Pattern 8: 3 tek + 3 çift
    def pattern_three_odd_three_even(self, k=20):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        
        top_odds = [num for num, _ in odd_counter.most_common(10)]
        top_evens = [num for num, _ in even_counter.most_common(10)]
        
        return (top_odds + top_evens)[:k]
    
    # Pattern 9: Hot numbers
    def pattern_hot(self, k=20):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 10: Sadece tek sayılar
    def pattern_only_odd(self, k=20):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        counter = Counter(odds)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 11: Son 30 çekiliş
    def pattern_son30(self, k=20):
        recent_nums = []
        window = min(30, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 12: Trend artan
    def pattern_trend_up(self, k=20, window=20):
        scores = {}
        for num in range(1, 61):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_numbers(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_numbers(row))
            
            if older_count > 0:
                trend = recent_count - older_count
            else:
                trend = recent_count
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # GEMINI 1: En çok çıkan sayı çiftleri
    def pattern_top_pairs(self, k=20):
        pair_count = defaultdict(int)
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for pair in combinations(nums, 2):
                pair_count[pair] += 1
        
        top_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:30]
        pair_numbers = []
        for pair, _ in top_pairs:
            pair_numbers.extend(pair)
        
        counter = Counter(pair_numbers)
        return [num for num, _ in counter.most_common(k)]
    
    # GEMINI 2: Ritmik Due analizi
    def pattern_rhythmic_due(self, k=20):
        last_seen = {num: 0 for num in range(1, 61)}
        intervals = {num: [] for num in range(1, 61)}
        last_pos = {num: -1 for num in range(1, 61)}
        
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                if last_pos[num] != -1:
                    intervals[num].append(idx - last_pos[num])
                last_pos[num] = idx
                last_seen[num] = idx
        
        last_idx = len(self.df) - 1
        
        rhythm_scores = {}
        for num in range(1, 61):
            if len(intervals[num]) >= 3:
                recent_intervals = intervals[num][-3:]
                if len(set(recent_intervals)) == 1:
                    rhythm_scores[num] = 2.0
                elif recent_intervals[-1] > recent_intervals[-2] > recent_intervals[-3]:
                    rhythm_scores[num] = 1.5
                else:
                    rhythm_scores[num] = 1.0
            else:
                rhythm_scores[num] = 1.0
        
        # Gecikme faktörü
        for num in range(1, 61):
            gap = last_idx - last_seen[num]
            if gap > 10:
                rhythm_scores[num] = rhythm_scores.get(num, 1.0) * (1 + gap/50)
        
        sorted_nums = sorted(rhythm_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # GEMINI 3: Yüksek entropi
    def pattern_high_entropy(self, k=20):
        def calculate_entropy(numbers):
            if not numbers:
                return 0
            ranges = [0] * 6
            for num in numbers:
                idx = (num - 1) // 10
                if 0 <= idx < 6:
                    ranges[idx] += 1
            entropy = 0
            total = len(numbers)
            for count in ranges:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            return entropy
        
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if calculate_entropy(nums) > 2.3:
                all_nums.extend(nums)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# KOMBİNASYON ÜRETİCİ
# ============================================================

class CombinationGenerator:
    def __init__(self, pool_numbers, pattern_weights, pattern_results):
        self.pool_numbers = pool_numbers
        self.pattern_weights = pattern_weights
        self.pattern_results = pattern_results
        self.number_scores = self._calculate_number_scores()
        
    def _calculate_number_scores(self):
        """Her sayının hangi pattern'lerde olduğuna göre puan hesapla"""
        scores = {}
        for num in self.pool_numbers:
            total_score = 0
            pattern_count = 0
            for pattern_key, numbers in self.pattern_results.items():
                if num in numbers:
                    weight = self.pattern_weights[pattern_key]['weight']
                    total_score += weight
                    pattern_count += 1
            scores[num] = {
                'total_score': total_score,
                'pattern_count': pattern_count
            }
        return scores
    
    def generate_combinations(self, num_combinations=100):
        """100 adet optimized kombinasyon üret"""
        combinations_list = []
        
        for combo_id in range(num_combinations):
            # Farklı stratejiler kullan
            strategy = combo_id % 5
            
            if strategy == 0:
                # Strateji 1: En yüksek puanlı sayılar
                selected = self._select_top_scored()
            elif strategy == 1:
                # Strateji 2: Pattern çeşitliliği
                selected = self._select_diverse_patterns()
            elif strategy == 2:
                # Strateji 3: Tek/Çift dengesi
                selected = self._select_balanced_odd_even()
            elif strategy == 3:
                # Strateji 4: Büyük/Küçük dengesi
                selected = self._select_balanced_small_large()
            else:
                # Strateji 5: Rastgele ağırlıklı
                selected = self._select_weighted_random()
            
            # Sırala ve kaydet
            selected_sorted = sorted(selected)
            combo_score = sum(self.number_scores[n]['total_score'] for n in selected)
            pattern_count_sum = sum(self.number_scores[n]['pattern_count'] for n in selected)
            
            combinations_list.append({
                'id': combo_id + 1,
                'numbers': selected_sorted,
                'strategy': strategy,
                'total_score': round(combo_score, 4),
                'avg_pattern_count': round(pattern_count_sum / 6, 2)
            })
        
        # Skora göre sırala
        combinations_list.sort(key=lambda x: x['total_score'], reverse=True)
        return combinations_list
    
    def _select_top_scored(self):
        """En yüksek puana sahip 6 sayı"""
        sorted_nums = sorted(self.number_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        return [num for num, _ in sorted_nums[:6]]
    
    def _select_diverse_patterns(self):
        """Farklı pattern'lerden sayı seç (çeşitlilik)"""
        selected = set()
        patterns_used = set()
        
        # Her pattern'den en yüksek puanlı sayıyı al
        for pattern_key in self.pattern_results.keys():
            if len(selected) >= 6:
                break
            for num in self.pattern_results[pattern_key]:
                if num not in selected and num in self.pool_numbers:
                    selected.add(num)
                    patterns_used.add(pattern_key)
                    break
        
        # Eksik varsa yüksek puanlılarla tamamla
        if len(selected) < 6:
            remaining = [n for n in self.pool_numbers if n not in selected]
            remaining_sorted = sorted(remaining, key=lambda x: self.number_scores[x]['total_score'], reverse=True)
            for num in remaining_sorted:
                if len(selected) >= 6:
                    break
                selected.add(num)
        
        return list(selected)
    
    def _select_balanced_odd_even(self):
        """3 tek + 3 çift dengesi"""
        odds = [n for n in self.pool_numbers if n % 2 == 1]
        evens = [n for n in self.pool_numbers if n % 2 == 0]
        
        odds_sorted = sorted(odds, key=lambda x: self.number_scores[x]['total_score'], reverse=True)
        evens_sorted = sorted(evens, key=lambda x: self.number_scores[x]['total_score'], reverse=True)
        
        selected = []
        selected.extend(odds_sorted[:3])
        selected.extend(evens_sorted[:3])
        
        return selected
    
    def _select_balanced_small_large(self):
        """3 küçük + 3 büyük dengesi"""
        small = [n for n in self.pool_numbers if n <= 30]
        large = [n for n in self.pool_numbers if n > 30]
        
        small_sorted = sorted(small, key=lambda x: self.number_scores[x]['total_score'], reverse=True)
        large_sorted = sorted(large, key=lambda x: self.number_scores[x]['total_score'], reverse=True)
        
        selected = []
        selected.extend(small_sorted[:3])
        selected.extend(large_sorted[:3])
        
        return selected
    
    def _select_weighted_random(self):
        """Ağırlıklı rastgele seçim"""
        weights = [self.number_scores[n]['total_score'] for n in self.pool_numbers]
        min_weight = min(weights)
        if min_weight <= 0:
            weights = [w + 0.1 for w in weights]
        
        selected = set()
        pool_copy = self.pool_numbers.copy()
        weights_copy = weights.copy()
        
        while len(selected) < 6 and pool_copy:
            # Ağırlıklı seçim
            chosen = random.choices(pool_copy, weights=weights_copy, k=1)[0]
            selected.add(chosen)
            
            # Seçilen sayıyı havuzdan çıkar
            idx = pool_copy.index(chosen)
            pool_copy.pop(idx)
            weights_copy.pop(idx)
        
        return list(selected)


# ============================================================
# ENSEMBLE BOT V2
# ============================================================

class EnsembleBotV2:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.target_pool_size = 22
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def get_ensemble_predictions(self):
        """Sadece 1.30+ pattern'leri kullan"""
        
        patterns = HighPerformancePatterns(self.df, self.get_numbers)
        
        # Tüm pattern'leri çalıştır
        pattern_results = {
            'odd_draws': patterns.pattern_odd_draws(20),
            'low_entropy': patterns.pattern_low_entropy(20),
            'son7': patterns.pattern_son7(20),
            'four_odd_two_even': patterns.pattern_4odd_2even(20),
            'son5': patterns.pattern_son5(20),
            'overdue': patterns.pattern_overdue(20),
            'three_small_three_large': patterns.pattern_three_small_three_large(20),
            'three_odd_three_even': patterns.pattern_three_odd_three_even(20),
            'hot': patterns.pattern_hot(20),
            'only_odd': patterns.pattern_only_odd(20),
            'son30': patterns.pattern_son30(20),
            'trend_up': patterns.pattern_trend_up(20),
            'top_pairs': patterns.pattern_top_pairs(20),
            'rhythmic_due': patterns.pattern_rhythmic_due(20),
            'high_entropy': patterns.pattern_high_entropy(20),
        }
        
        # Ağırlıklı puan hesapla
        scores = {}
        pattern_details = {}
        
        for pattern_key, numbers in pattern_results.items():
            weight = patterns.patterns_info[pattern_key]['weight']
            score_val = patterns.patterns_info[pattern_key]['score']
            
            for num in numbers:
                scores[num] = scores.get(num, 0) + weight
                if num not in pattern_details:
                    pattern_details[num] = []
                pattern_details[num].append({
                    'pattern': patterns.patterns_info[pattern_key]['name'],
                    'weight': weight,
                    'score': score_val
                })
        
        # Puana göre sırala
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Havuz büyüklüğünü belirle
        pool_size = self.target_pool_size
        if len(sorted_numbers) < pool_size:
            pool_size = len(sorted_numbers)
        
        pool_numbers = [num for num, _ in sorted_numbers[:pool_size]]
        
        # Detaylı rapor
        detailed = []
        for num, weighted_score in sorted_numbers[:pool_size]:
            patterns_list = [p['pattern'] for p in pattern_details[num]]
            pattern_scores = [p['score'] for p in pattern_details[num]]
            
            detailed.append({
                'number': num,
                'weighted_score': round(weighted_score, 4),
                'pattern_count': len(patterns_list),
                'patterns': patterns_list,
                'avg_pattern_score': round(sum(pattern_scores) / len(pattern_scores), 2)
            })
        
        # Kombinasyon üret
        combo_gen = CombinationGenerator(pool_numbers, patterns.patterns_info, pattern_results)
        combinations = combo_gen.generate_combinations(100)
        
        # Top 12 ve Top 6
        top_12 = [item['number'] for item in detailed[:12]]
        top_6 = [item['number'] for item in detailed[:6]]
        
        return detailed, top_12, top_6, combinations, pattern_results, patterns.patterns_info
    
    def print_report(self):
        detailed, top_12, top_6, combinations, pattern_results, patterns_info = self.get_ensemble_predictions()
        
        print("\n" + "=" * 80)
        print("🎯 SÜPER LOTO ENSEMBLE BOT V2 (Gemini Optimizasyonlu)")
        print("   SADECE 1.30+ BAŞARILI PATTERN'LER")
        print(f"   Toplam {len(patterns_info)} pattern | Hedef havuz: 20-25 sayı")
        print("   100 ADET OPTİMİZE KOMBİNASYON ÜRETİLDİ")
        print("=" * 80)
        
        print("\n📊 KULLANILAN PATTERN'LER (V2 backtest sonuçları):")
        print("-" * 80)
        for key, info in sorted(patterns_info.items(), key=lambda x: x[1]['score'], reverse=True):
            gemini_tag = " [Gemini]" if key in ['top_pairs', 'rhythmic_due', 'high_entropy'] else ""
            print(f"  {info['name']:40} | Başarı: {info['score']:.2f}/12 | Ağırlık: %{info['weight']*100:.1f}{gemini_tag}")
        
        print("\n" + "-" * 80)
        print(f"🏆 SAYI HAVUZU ({len(detailed)} sayı)")
        print("-" * 80)
        print(f"\n  {'Sayı':>5} | {'Ağırlık':>8} | {'Pattern Sayısı':>13} | {'Ort. Pattern Puanı':>18} | {'Patternler'}")
        print("  " + "-" * 78)
        
        for item in detailed[:25]:
            patterns_short = ', '.join(item['patterns'][:2])
            if len(item['patterns']) > 2:
                patterns_short += f" +{len(item['patterns'])-2}"
            print(f"  {item['number']:5d} | {item['weighted_score']:8.4f} | {item['pattern_count']:13d} | {item['avg_pattern_score']:18.2f} | {patterns_short}")
        
        print("\n" + "-" * 80)
        print("🎯 ÖNERİLEN 12 SAYI (En yüksek ağırlıklı puan)")
        print("-" * 80)
        
        for i in range(0, len(top_12), 4):
            group = top_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 80)
        print("🔥 ÖNERİLEN 6 SAYI (En güçlü)")
        print("-" * 80)
        print(f"\n  🌟🌟🌟  {top_6}  🌟🌟🌟")
        
        # En iyi 10 kombinasyon
        print("\n" + "-" * 80)
        print("🎲 EN İYİ 10 KOMBİNASYON (Toplam 100 üretildi)")
        print("-" * 80)
        
        strategies = {
            0: "En yüksek puanlı",
            1: "Pattern çeşitliliği",
            2: "3 tek + 3 çift",
            3: "3 küçük + 3 büyük",
            4: "Ağırlıklı rastgele"
        }
        
        for combo in combinations[:10]:
            print(f"  #{combo['id']:3d}: {combo['numbers']} | Skor: {combo['total_score']:.4f} | {strategies[combo['strategy']]} | Ort.Pattern: {combo['avg_pattern_count']}")
        
        print(f"\n  ... ve {len(combinations)-10} kombinasyon daha (TXT dosyasında tam liste)")
        
        # Havuz istatistiği
        print("\n" + "-" * 80)
        print("📊 HAVUZ İSTATİSTİKLERİ")
        print("-" * 80)
        
        all_numbers = [item['number'] for item in detailed]
        
        odds = [n for n in all_numbers if n % 2 == 1]
        evens = [n for n in all_numbers if n % 2 == 0]
        print(f"  Tek sayılar: {len(odds)} ({len(odds)/len(all_numbers)*100:.0f}%)")
        print(f"  Çift sayılar: {len(evens)} ({len(evens)/len(all_numbers)*100:.0f}%)")
        
        small = [n for n in all_numbers if n <= 30]
        large = [n for n in all_numbers if n > 30]
        print(f"  Küçük sayılar (1-30): {len(small)} ({len(small)/len(all_numbers)*100:.0f}%)")
        print(f"  Büyük sayılar (31-60): {len(large)} ({len(large)/len(all_numbers)*100:.0f}%)")
        
        print("\n" + "-" * 80)
        print("⚠️ NOT: Gemini önerileri (Pairwise, Rhythmic Due, High Entropy) eklendi.")
        print("   100 kombinasyon outputs/ensemble_v2_combinations.txt dosyasında")
        print("=" * 80)
        
        return {
            'pool_size': len(detailed),
            'top_12': top_12,
            'top_6': top_6,
            'combinations': combinations,
            'all_numbers': detailed,
            'patterns_used': len(patterns_info)
        }
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        # JSON kaydet
        with open('outputs/ensemble_v2_predictions.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'V2_Gemini',
                'pool_size': result['pool_size'],
                'recommended_12_numbers': result['top_12'],
                'recommended_6_numbers': result['top_6'],
                'combinations_count': len(result['combinations']),
                'all_numbers_with_scores': result['all_numbers'],
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        
        # TXT dosyası - 100 kombinasyon
        with open('outputs/ensemble_v2_combinations.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🎯 SÜPER LOTO ENSEMBLE BOT V2 - 100 KOMBİNASYON\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Oluşturulma Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Havuz Büyüklüğü: {result['pool_size']} sayı\n")
            f.write(f"Toplam Kombinasyon: {len(result['combinations'])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("📊 HAVUZDAKİ SAYILAR (Ağırlıklı puan sıralı)\n")
            f.write("-" * 80 + "\n")
            for item in result['all_numbers']:
                f.write(f"  {item['number']:3d}: {item['weighted_score']:.4f} puan ({item['pattern_count']} pattern)\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("🎲 100 KOMBİNASYON (Skora göre sıralı)\n")
            f.write("-" * 80 + "\n\n")
            
            strategies = {
                0: "En yüksek puanlı",
                1: "Pattern çeşitliliği",
                2: "3 tek + 3 çift",
                3: "3 küçük + 3 büyük",
                4: "Ağırlıklı rastgele"
            }
            
            for combo in result['combinations']:
                f.write(f"#{combo['id']:3d} | {combo['numbers']} | ")
                f.write(f"Skor: {combo['total_score']:.4f} | ")
                f.write(f"Strateji: {strategies[combo['strategy']]} | ")
                f.write(f"Ort.Pattern: {combo['avg_pattern_count']}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("🎯 EN İYİ 12 SAYI (Havuzdan)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {result['top_12']}\n\n")
            f.write("🔥 EN İYİ 6 SAYI\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {result['top_6']}\n")
        
        # Rapor TXT
        with open('outputs/ensemble_v2_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🎯 SÜPER LOTO ENSEMBLE BOT V2 RAPORU (Gemini Optimizasyonlu)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Havuz Büyüklüğü: {result['pool_size']} sayı\n")
            f.write(f"Üretilen Kombinasyon: {len(result['combinations'])}\n\n")
            f.write("ÖNERİLEN 12 SAYI:\n")
            f.write(str(result['top_12']) + "\n\n")
            f.write("ÖNERİLEN 6 SAYI:\n")
            f.write(str(result['top_6']) + "\n\n")
            f.write("TÜM KOMBİNASYONLAR:\n")
            f.write("  ensemble_v2_combinations.txt dosyasında\n")
        
        print(f"\n💾 Kaydedildi: outputs/ensemble_v2_predictions.json")
        print(f"💾 Kaydedildi: outputs/ensemble_v2_combinations.txt (100 kombinasyon)")
        print(f"💾 Kaydedildi: outputs/ensemble_v2_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 SÜPER LOTO ENSEMBLE BOT V2 (Gemini Optimizasyonlu)")
    print("   SADECE 1.30+ BAŞARILI PATTERN'LER")
    print(f"   15 pattern kullanılıyor (3 Gemini önerisi eklendi)")
    print("   100 ADET OPTİMİZE KOMBİNASYON ÜRETİLECEK")
    print("=" * 80)
    
    bot = EnsembleBotV2()
    bot.load_data()
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
