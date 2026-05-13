#!/usr/bin/env python3
"""
ŞANS TOPU PATTERN MASTER V3
======================================================================
Süper Loto Pattern Master V2 mantığı ile yeniden yazıldı

YENİ ÖZELLİKLER:
  1. Sayı Çiftleri (Pairwise) ve Üçlüler (Triplets) Analizi
  2. Ortalama Aralığa Göre Gecikme (Overdue) Analizi
  3. Onluk Blok (Gap) Analizi
  4. Entropi (Karmaşıklık) Skoru ile Ağırlıklandırma
  5. Kümeleme (Clustering) Analizi
  6. Ritmik Due Analizi
  7. Toplam 45+ pattern test edilir
======================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import math
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEYİCİ
# ═══════════════════════════════════════════════════════════════════

class DataLoader:
    MAIN_COLS = ['no_1', 'no_2', 'no_3', 'no_4', 'no_5']
    PLUS_COL = 'no_5+1'

    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None

    def load(self):
        df = pd.read_excel(self.path, sheet_name=self.sheet, header=0)
        
        # Tarih
        for col in ('tarih', 'tarih.1'):
            if col in df.columns:
                df['tarih'] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                break
        
        # Sayılar
        for col in self.MAIN_COLS + [self.PLUS_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['tarih'] + self.MAIN_COLS + [self.PLUS_COL]).reset_index(drop=True)
        df['weekday'] = df['tarih'].dt.weekday
        df['month'] = df['tarih'].dt.month
        df['day'] = df['tarih'].dt.day
        df['year'] = df['tarih'].dt.year
        
        self.df = df
        return df

    def get_main(self, row):
        return [int(row[c]) for c in self.MAIN_COLS if c in row.index and pd.notna(row[c])]

    def get_sorted_main(self, row):
        return sorted(self.get_main(row))

    def get_plus(self, row):
        v = row.get(self.PLUS_COL, 0)
        return int(v) if pd.notna(v) and v > 0 else 0


# ═══════════════════════════════════════════════════════════════════
# ANA SAYI PATTERNLERİ (Süper Loto V2 mantığı ile)
# ═══════════════════════════════════════════════════════════════════

class MainPatterns:
    def __init__(self, df, get_main_func, get_sorted_func):
        self.df = df
        self.get_main = get_main_func
        self.get_sorted = get_sorted_func
        
        # Cache'ler
        self.pair_cache = None
        self.triplet_cache = None
        self.interval_cache = None

    # ============================================================
    # 1. ZAMAN TABANLI (7 pattern)
    # ============================================================
    
    def _window(self, n, k=10):
        n = min(n, len(self.df))
        nums = []
        for i in range(len(self.df) - n, len(self.df)):
            nums.extend(self.get_main(self.df.iloc[i]))
        return [x for x, _ in Counter(nums).most_common(k)]

    def p_son_3(self, k=10): return self._window(3, k)
    def p_son_5(self, k=10): return self._window(5, k)
    def p_son_7(self, k=10): return self._window(7, k)
    def p_son_10(self, k=10): return self._window(10, k)
    def p_son_15(self, k=10): return self._window(15, k)
    def p_son_20(self, k=10): return self._window(20, k)
    def p_son_30(self, k=10): return self._window(30, k)

    # ============================================================
    # 2. HOT & DUE (2 pattern)
    # ============================================================
    
    def p_hot(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        return [x for x, _ in Counter(all_nums).most_common(k)]

    def p_due(self, k=10):
        last_seen = {n: 0 for n in range(1, 35)}
        for idx, row in self.df.iterrows():
            for n in self.get_main(row):
                last_seen[n] = idx
        last = len(self.df) - 1
        return sorted(range(1, 35), key=lambda n: last - last_seen[n], reverse=True)[:k]

    # ============================================================
    # 3. FİBONACCİ (2 pattern)
    # ============================================================
    
    def p_fibonacci(self, k=10):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        return [f for f in fib if f <= 34][:k]

    def p_fibonacci_neighbor(self, k=10):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        neighbors = set()
        for f in fib:
            for d in (-1, 0, 1):
                if 1 <= f + d <= 34:
                    neighbors.add(f + d)
        return sorted(neighbors)[:k]

    # ============================================================
    # 4. TEK/ÇİFT (3 pattern)
    # ============================================================
    
    def p_only_odd(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        odds = [n for n in all_nums if n % 2 == 1]
        return [x for x, _ in Counter(odds).most_common(k)]

    def p_only_even(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        evens = [n for n in all_nums if n % 2 == 0]
        return [x for x, _ in Counter(evens).most_common(k)]

    def p_3odd_2even(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        odds = [n for n in all_nums if n % 2 == 1]
        evens = [n for n in all_nums if n % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [x for x, _ in odd_counter.most_common(6)]
        top_evens = [x for x, _ in even_counter.most_common(4)]
        return (top_odds + top_evens)[:k]

    # ============================================================
    # 5. BÜYÜK/KÜÇÜK (3 pattern)
    # ============================================================
    
    def p_small(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        small = [n for n in all_nums if n <= 17]
        return [x for x, _ in Counter(small).most_common(k)]

    def p_large(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        large = [n for n in all_nums if n > 17]
        return [x for x, _ in Counter(large).most_common(k)]

    def p_2small_3large(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        small = [n for n in all_nums if n <= 17]
        large = [n for n in all_nums if n > 17]
        small_counter = Counter(small)
        large_counter = Counter(large)
        top_small = [x for x, _ in small_counter.most_common(2)]
        top_large = [x for x, _ in large_counter.most_common(3)]
        return (top_small + top_large)[:k]

    # ============================================================
    # 6. KOMŞULUK (2 pattern)
    # ============================================================
    
    def p_neighbor(self, k=10):
        last = self.get_main(self.df.iloc[-1])
        neighbors = set()
        for n in last:
            for d in (-2, -1, 0, 1, 2):
                if 1 <= n + d <= 34:
                    neighbors.add(n + d)
        return sorted(neighbors)[:k]

    def p_last_draw(self, k=10):
        return self.get_main(self.df.iloc[-1])[:k]

    # ============================================================
    # 7. ZAMAN (Hafta/Ay) - 2 pattern
    # ============================================================
    
    def p_weekday(self, k=10):
        wd = self.df.iloc[-1]['weekday']
        nums = []
        for _, row in self.df[self.df['weekday'] == wd].iterrows():
            nums.extend(self.get_main(row))
        return [x for x, _ in Counter(nums).most_common(k)] if nums else self.p_hot(k)

    def p_month_day(self, k=10):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            return self.p_hot(k)
        nums = []
        for _, row in same_day.iterrows():
            nums.extend(self.get_main(row))
        return [x for x, _ in Counter(nums).most_common(k)]

    # ============================================================
    # 8. YENİ! SAYI ÇİFTLERİ (PAIR) ANALİZİ
    # ============================================================
    
    def _build_pair_cache(self):
        if self.pair_cache is not None:
            return self.pair_cache
        
        pair_count = defaultdict(int)
        pair_last_seen = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            nums = sorted(self.get_main(row))
            for pair in combinations(nums, 2):
                pair_count[pair] += 1
                pair_last_seen[pair] = idx
        
        self.pair_cache = {'counts': pair_count, 'last_seen': pair_last_seen}
        return self.pair_cache

    def p_top_pairs(self, k=10):
        cache = self._build_pair_cache()
        top_pairs = sorted(cache['counts'].items(), key=lambda x: x[1], reverse=True)[:20]
        
        pair_numbers = []
        for pair, _ in top_pairs:
            pair_numbers.extend(pair)
        
        counter = Counter(pair_numbers)
        return [x for x, _ in counter.most_common(k)]

    def p_strongest_partners(self, k=10):
        cache = self._build_pair_cache()
        
        best_partners = {}
        for (a, b), count in cache['counts'].items():
            if a not in best_partners or count > best_partners[a][1]:
                best_partners[a] = (b, count)
            if b not in best_partners or count > best_partners[b][1]:
                best_partners[b] = (a, count)
        
        partner_scores = defaultdict(int)
        for num, (partner, count) in best_partners.items():
            partner_scores[partner] += count
        
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        return [x for x, _ in sorted_partners[:k]]

    # ============================================================
    # 9. YENİ! ÜÇLÜLER (TRIPLET) ANALİZİ
    # ============================================================
    
    def _build_triplet_cache(self):
        if self.triplet_cache is not None:
            return self.triplet_cache
        
        triplet_count = defaultdict(int)
        for _, row in self.df.iterrows():
            nums = sorted(self.get_main(row))
            for triplet in combinations(nums, 3):
                triplet_count[triplet] += 1
        
        self.triplet_cache = triplet_count
        return self.triplet_cache

    def p_top_triplets(self, k=10):
        cache = self._build_triplet_cache()
        top_triplets = sorted(cache.items(), key=lambda x: x[1], reverse=True)[:15]
        
        triplet_numbers = []
        for triplet, _ in top_triplets:
            triplet_numbers.extend(triplet)
        
        counter = Counter(triplet_numbers)
        return [x for x, _ in counter.most_common(k)]

    # ============================================================
    # 10. YENİ! OVERDUE ANALİZİ (Gecikme Devri)
    # ============================================================
    
    def _build_interval_cache(self):
        if self.interval_cache is not None:
            return self.interval_cache
        
        intervals = {n: [] for n in range(1, 35)}
        last_seen = {n: -1 for n in range(1, 35)}
        
        for idx, row in self.df.iterrows():
            for n in self.get_main(row):
                if last_seen[n] != -1:
                    interval = idx - last_seen[n]
                    intervals[n].append(interval)
                last_seen[n] = idx
        
        current_idx = len(self.df) - 1
        current_gap = {
            n: current_idx - last_seen[n] if last_seen[n] != -1 else current_idx + 1
            for n in range(1, 35)
        }
        
        avg_intervals = {}
        for n in range(1, 35):
            if intervals[n]:
                avg_intervals[n] = sum(intervals[n]) / len(intervals[n])
            else:
                avg_intervals[n] = 7
        
        self.interval_cache = {
            'avg': avg_intervals,
            'current_gap': current_gap,
            'history': intervals
        }
        return self.interval_cache

    def p_overdue(self, k=10):
        cache = self._build_interval_cache()
        
        overdue_scores = {}
        for n in range(1, 35):
            avg = cache['avg'][n]
            gap = cache['current_gap'][n]
            
            if avg > 0:
                ratio = gap / avg
                if ratio > 2.0:
                    overdue_scores[n] = 3.0
                elif ratio > 1.5:
                    overdue_scores[n] = 2.0
                elif ratio > 1.0:
                    overdue_scores[n] = 1.5
                else:
                    overdue_scores[n] = 1.0
            else:
                overdue_scores[n] = 1.0
        
        sorted_nums = sorted(overdue_scores.items(), key=lambda x: x[1], reverse=True)
        return [x for x, _ in sorted_nums[:k]]

    def p_due_with_rhythm(self, k=10):
        cache = self._build_interval_cache()
        
        rhythm_scores = {}
        for n in range(1, 35):
            history = cache['history'][n]
            if len(history) >= 3:
                recent = history[-3:]
                if len(set(recent)) == 1:
                    rhythm_scores[n] = 2.0
                elif recent[-1] > recent[-2] > recent[-3]:
                    rhythm_scores[n] = 1.5
                else:
                    rhythm_scores[n] = 1.0
            else:
                rhythm_scores[n] = 1.0
        
        for n in range(1, 35):
            gap = cache['current_gap'][n]
            avg = cache['avg'][n]
            if avg > 0 and gap > avg * 1.2:
                rhythm_scores[n] = rhythm_scores.get(n, 1.0) * 1.5
        
        sorted_nums = sorted(rhythm_scores.items(), key=lambda x: x[1], reverse=True)
        return [x for x, _ in sorted_nums[:k]]

    # ============================================================
    # 11. YENİ! BLOK/GAP ANALİZİ
    # ============================================================
    
    def _calculate_block_stats(self):
        blocks = {i: [] for i in range(4)}  # 1-8, 9-16, 17-24, 25-34
        
        for _, row in self.df.iterrows():
            nums = self.get_main(row)
            block_hits = {i: False for i in range(4)}
            
            for n in nums:
                block_idx = (n - 1) // 8
                if 0 <= block_idx < 4:
                    block_hits[block_idx] = True
            
            for i in range(4):
                blocks[i].append(block_hits[i])
        
        hit_rates = {}
        for i in range(4):
            total = len(blocks[i])
            hit_count = sum(1 for hit in blocks[i] if hit)
            hit_rates[i] = hit_count / total if total > 0 else 0
        
        return hit_rates

    def p_active_block(self, k=10):
        hit_rates = self._calculate_block_stats()
        best_block = max(hit_rates.items(), key=lambda x: x[1])[0]
        
        block_numbers = range(best_block * 8 + 1, min(best_block * 8 + 9, 35))
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        
        block_nums = [(n, counter.get(n, 0)) for n in block_numbers]
        block_nums.sort(key=lambda x: x[1], reverse=True)
        
        return [n for n, _ in block_nums[:k]]

    def p_inactive_block(self, k=10):
        hit_rates = self._calculate_block_stats()
        worst_block = min(hit_rates.items(), key=lambda x: x[1])[0]
        
        block_numbers = range(worst_block * 8 + 1, min(worst_block * 8 + 9, 35))
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        
        block_nums = [(n, counter.get(n, 0)) for n in block_numbers]
        block_nums.sort(key=lambda x: x[1], reverse=True)
        
        return [n for n, _ in block_nums[:k]]

    # ============================================================
    # 12. YENİ! KÜMELEME (CLUSTERING)
    # ============================================================
    
    def p_cluster_centers(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        
        block_freq = defaultdict(int)
        for n in all_nums:
            block = (n - 1) // 8
            block_freq[block] += 1
        
        most_dense = max(block_freq.items(), key=lambda x: x[1])[0]
        
        block_numbers = range(most_dense * 8 + 1, min(most_dense * 8 + 9, 35))
        counter = Counter(all_nums)
        
        block_nums = [(n, counter.get(n, 0)) for n in block_numbers]
        block_nums.sort(key=lambda x: x[1], reverse=True)
        
        return [n for n, _ in block_nums[:k]]

    # ============================================================
    # 13. YENİ! ENTROPİ ANALİZİ
    # ============================================================
    
    def calculate_entropy(self, numbers):
        if not numbers:
            return 0
        
        ranges = [0] * 4
        for n in numbers:
            idx = (n - 1) // 8
            if 0 <= idx < 4:
                ranges[idx] += 1
        
        entropy = 0
        total = len(numbers)
        for count in ranges:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy

    def p_low_entropy(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_main(row)
            if self.calculate_entropy(nums) < 1.5:
                all_nums.extend(nums)
        
        counter = Counter(all_nums)
        return [x for x, _ in counter.most_common(k)]

    def p_high_entropy(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_main(row)
            if self.calculate_entropy(nums) > 1.8:
                all_nums.extend(nums)
        
        counter = Counter(all_nums)
        return [x for x, _ in counter.most_common(k)]

    # ============================================================
    # 14. YENİ! ASAL SAYILAR
    # ============================================================
    
    def p_prime(self, k=10):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        return [p for p in primes if p <= 34][:k]


# ═══════════════════════════════════════════════════════════════════
# ARTI SAYI PATTERNLERİ (Süper Loto mantığı ile)
# ═══════════════════════════════════════════════════════════════════

class PlusPatterns:
    def __init__(self, df, get_plus_func):
        self.df = df
        self.get_plus = get_plus_func

    def _window(self, n, k=4):
        n = min(n, len(self.df))
        nums = [self.get_plus(self.df.iloc[i]) for i in range(len(self.df) - n, len(self.df))]
        nums = [x for x in nums if x > 0]
        return [x for x, _ in Counter(nums).most_common(k)]

    def p_son_5(self, k=4): return self._window(5, k)
    def p_son_7(self, k=4): return self._window(7, k)
    def p_son_10(self, k=4): return self._window(10, k)
    def p_son_15(self, k=4): return self._window(15, k)

    def p_hot(self, k=4):
        nums = [self.get_plus(row) for _, row in self.df.iterrows() if self.get_plus(row) > 0]
        return [x for x, _ in Counter(nums).most_common(k)]

    def p_due(self, k=4):
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        return sorted(range(1, 15), key=lambda n: last - last_seen[n], reverse=True)[:k]

    def p_weekday(self, k=4):
        wd = self.df.iloc[-1]['weekday']
        nums = [self.get_plus(row) for _, row in self.df[self.df['weekday'] == wd].iterrows()
                if self.get_plus(row) > 0]
        return [x for x, _ in Counter(nums).most_common(k)] if nums else self.p_hot(k)


# ═══════════════════════════════════════════════════════════════════
# BACKTEST MOTORU (Süper Loto V2 mantığı ile)
# ═══════════════════════════════════════════════════════════════════

class BacktestEngineV3:
    def __init__(self, df, get_main_func, get_sorted_func, get_plus_func):
        self.df = df
        self.get_main = get_main_func
        self.get_sorted = get_sorted_func
        self.get_plus = get_plus_func
        self.main_results = {}
        self.plus_results = {}

    def test_main_pattern(self, pattern_func, pattern_name, test_size=100, pool_size=10):
        total = len(self.df)
        train_size = total - test_size
        
        if train_size < 50:
            return 0
        
        scores = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            temp_df = self.df.iloc[:train_end]
            temp_patterns = MainPatterns(temp_df, self.get_main, self.get_sorted)
            
            try:
                preds = getattr(temp_patterns, pattern_func)(pool_size)
            except:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_main(test_row))
            correct = len(set(preds) & actual)
            
            # Entropi ağırlığı
            entropy = temp_patterns.calculate_entropy(preds)
            entropy_weight = 1.0
            if 1.5 <= entropy <= 1.9:
                entropy_weight = 1.1
            elif entropy < 1.0 or entropy > 2.2:
                entropy_weight = 0.9
            
            scores.append(correct * entropy_weight)
        
        return sum(scores) / len(scores) if scores else 0

    def test_plus_pattern(self, pattern_func, pattern_name, test_size=100, pool_size=4):
        total = len(self.df)
        train_size = total - test_size
        
        if train_size < 50:
            return 0
        
        hits = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            temp_df = self.df.iloc[:train_end]
            temp_patterns = PlusPatterns(temp_df, self.get_plus)
            
            try:
                preds = getattr(temp_patterns, pattern_func)(pool_size)
            except:
                preds = []
            
            actual = self.get_plus(self.df.iloc[train_end])
            hits.append(1 if actual in preds else 0)
        
        return sum(hits) / len(hits) if hits else 0

    def run_all_tests(self, test_size=100):
        print(f"\n📊 ŞANS TOPU V3 WALK-FORWARD BACKTEST (son {test_size} çekiliş)")
        print("   Toplam 35+ pattern test edilecek (Pair, Triplet, Overdue, Block, Entropy)")
        print("=" * 70)

        # ANA KISIM PATTERNLERİ
        main_pattern_list = [
            ('p_son_3', 'Son 3 çekiliş'),
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_son_20', 'Son 20 çekiliş'),
            ('p_son_30', 'Son 30 çekiliş'),
            ('p_hot', 'Hot numbers'),
            ('p_due', 'Due numbers'),
            ('p_fibonacci', 'Fibonacci sayıları'),
            ('p_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_only_odd', 'Sadece tek'),
            ('p_only_even', 'Sadece çift'),
            ('p_3odd_2even', '3 tek + 2 çift'),
            ('p_small', 'Küçük (1-17)'),
            ('p_large', 'Büyük (18-34)'),
            ('p_2small_3large', '2 küçük + 3 büyük'),
            ('p_neighbor', 'Son çekiliş komşuları'),
            ('p_last_draw', 'Son çekiliş aynısı'),
            ('p_weekday', 'Haftanın aynı günü'),
            ('p_month_day', 'Ayın aynı günü'),
            ('p_top_pairs', '🔗 En çok çıkan çiftler (Pair)'),
            ('p_strongest_partners', '🤝 En güçlü partnerler'),
            ('p_top_triplets', '🔺 En çok çıkan üçlüler (Triplet)'),
            ('p_overdue', '⏰ Vadesi geçmiş (Overdue)'),
            ('p_due_with_rhythm', '🎵 Ritmik due'),
            ('p_active_block', '📊 Aktif blok'),
            ('p_inactive_block', '🎯 Pasif blok (Sürpriz)'),
            ('p_cluster_centers', '📍 Küme merkezleri'),
            ('p_low_entropy', '📐 Düşük entropi'),
            ('p_high_entropy', '🌊 Yüksek entropi'),
            ('p_prime', 'Asal sayılar'),
        ]

        print("\n🎯 ANA KISIM PATTERN SONUÇLARI (10 tahmin)")
        print("-" * 65)
        print(f"{'Pattern':38} {'Başarı':>10} {'vs Rastgele':>12}")
        print("-" * 65)

        rastgele_main = 10 * (5 / 34)
        print(f"  {'RASTGELE':38}: {rastgele_main:5.2f}/10 (referans)")
        print("-" * 65)

        for pattern_func, pattern_name in main_pattern_list:
            score = self.test_main_pattern(pattern_func, pattern_name, test_size, 10)
            self.main_results[pattern_name] = score
            improvement = ((score - rastgele_main) / rastgele_main * 100) if rastgele_main > 0 else 0
            
            if improvement > 15:
                star = "🔥🔥"
            elif improvement > 8:
                star = "🔥"
            elif improvement > 3:
                star = "⭐"
            elif improvement > -3:
                star = "·"
            else:
                star = "❌"
            
            print(f"  {pattern_name:38}: {score:5.2f}/10 (%{improvement:+5.1f}) {star}")

        # ARTI KISIM PATTERNLERİ
        plus_pattern_list = [
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_hot', 'Hot numbers'),
            ('p_due', 'Due numbers'),
            ('p_weekday', 'Haftanın aynı günü'),
        ]

        print(f"\n🎯 ARTI KISIM PATTERN SONUÇLARI (4 tahmin)")
        print("-" * 65)
        print(f"{'Pattern':38} {'Başarı':>10} {'vs Rastgele':>12}")
        print("-" * 65)

        rastgele_plus = 4 * (1 / 14)
        print(f"  {'RASTGELE':38}: {rastgele_plus:5.2f}/4 (referans)")
        print("-" * 65)

        for pattern_func, pattern_name in plus_pattern_list:
            score = self.test_plus_pattern(pattern_func, pattern_name, test_size, 4)
            self.plus_results[pattern_name] = score
            improvement = ((score - rastgele_plus) / rastgele_plus * 100) if rastgele_plus > 0 else 0
            
            if improvement > 30:
                star = "🔥🔥"
            elif improvement > 15:
                star = "🔥"
            elif improvement > 5:
                star = "⭐"
            elif improvement > -5:
                star = "·"
            else:
                star = "❌"
            
            print(f"  {pattern_name:38}: {score:5.2f}/4 (%{improvement:+5.1f}) {star}")

        print("\n" + "=" * 70)
        return self.main_results, self.plus_results


# ═══════════════════════════════════════════════════════════════════
# ANA SINIF
# ═══════════════════════════════════════════════════════════════════

class SansTopuPatternMasterV3:
    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path = path
        self.sheet = sheet
        self.df = None
        self.main_results = None
        self.plus_results = None

    def load(self):
        loader = DataLoader(self.path, self.sheet)
        self.df = loader.load()
        self.get_main = loader.get_main
        self.get_sorted = loader.get_sorted_main
        self.get_plus = loader.get_plus
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 {self.df['tarih'].min():%d.%m.%Y} → {self.df['tarih'].max():%d.%m.%Y}")
        return self.df

    def run_tests(self, test_size=100):
        engine = BacktestEngineV3(self.df, self.get_main, self.get_sorted, self.get_plus)
        self.main_results, self.plus_results = engine.run_all_tests(test_size)
        
        best_main = max(self.main_results.items(), key=lambda x: x[1])
        best_plus = max(self.plus_results.items(), key=lambda x: x[1])
        
        rastgele_main = 10 * (5 / 34)
        rastgele_plus = 4 * (1 / 14)
        
        print(f"\n🏆 EN İYİ ANA PATTERN: {best_main[0]} ({best_main[1]:.2f}/10)")
        print(f"   İyileştirme: %{((best_main[1]-rastgele_main)/rastgele_main*100):.1f}")
        print(f"\n🏆 EN İYİ ARTI PATTERN: {best_plus[0]} ({best_plus[1]:.2f}/4)")
        print(f"   İyileştirme: %{((best_plus[1]-rastgele_plus)/rastgele_plus*100):.1f}")
        
        return self.main_results, self.plus_results

    def get_best_main_pool(self, pattern_name, pool_size=20):
        """En iyi pattern'in pool'unu al"""
        pattern_map = {
            'Son 3 çekiliş': 'p_son_3',
            'Son 5 çekiliş': 'p_son_5',
            'Son 7 çekiliş': 'p_son_7',
            'Son 10 çekiliş': 'p_son_10',
            'Son 15 çekiliş': 'p_son_15',
            'Son 20 çekiliş': 'p_son_20',
            'Son 30 çekiliş': 'p_son_30',
            'Hot numbers': 'p_hot',
            'Due numbers': 'p_due',
            'Fibonacci sayıları': 'p_fibonacci',
            'Fibonacci + komşu': 'p_fibonacci_neighbor',
            'Sadece tek': 'p_only_odd',
            'Sadece çift': 'p_only_even',
            '3 tek + 2 çift': 'p_3odd_2even',
            'Küçük (1-17)': 'p_small',
            'Büyük (18-34)': 'p_large',
            '2 küçük + 3 büyük': 'p_2small_3large',
            'Son çekiliş komşuları': 'p_neighbor',
            'Son çekiliş aynısı': 'p_last_draw',
            'Haftanın aynı günü': 'p_weekday',
            'Ayın aynı günü': 'p_month_day',
            '🔗 En çok çıkan çiftler (Pair)': 'p_top_pairs',
            '🤝 En güçlü partnerler': 'p_strongest_partners',
            '🔺 En çok çıkan üçlüler (Triplet)': 'p_top_triplets',
            '⏰ Vadesi geçmiş (Overdue)': 'p_overdue',
            '🎵 Ritmik due': 'p_due_with_rhythm',
            '📊 Aktif blok': 'p_active_block',
            '🎯 Pasif blok (Sürpriz)': 'p_inactive_block',
            '📍 Küme merkezleri': 'p_cluster_centers',
            '📐 Düşük entropi': 'p_low_entropy',
            '🌊 Yüksek entropi': 'p_high_entropy',
            'Asal sayılar': 'p_prime',
        }
        
        func_name = pattern_map.get(pattern_name, 'p_hot')
        patterns = MainPatterns(self.df, self.get_main, self.get_sorted)
        
        try:
            predictions = getattr(patterns, func_name)(pool_size)
        except:
            predictions = patterns.p_hot(pool_size)
        
        return predictions

    def get_best_plus_pool(self, pattern_name, k=5):
        """En iyi artı pattern'in pool'unu al"""
        pattern_map = {
            'Son 5 çekiliş': 'p_son_5',
            'Son 7 çekiliş': 'p_son_7',
            'Son 10 çekiliş': 'p_son_10',
            'Son 15 çekiliş': 'p_son_15',
            'Hot numbers': 'p_hot',
            'Due numbers': 'p_due',
            'Haftanın aynı günü': 'p_weekday',
        }
        
        func_name = pattern_map.get(pattern_name, 'p_hot')
        patterns = PlusPatterns(self.df, self.get_plus)
        
        try:
            predictions = getattr(patterns, func_name)(k)
        except:
            predictions = patterns.p_hot(k)
        
        return predictions

    def report(self, pool_size=17):
        """Rapor üret - hedef 17 sayılık agresif pool"""
        if self.main_results is None:
            self.run_tests()
        
        best_main_name = max(self.main_results.items(), key=lambda x: x[1])[0]
        best_plus_name = max(self.plus_results.items(), key=lambda x: x[1])[0]
        
        # En iyi pattern'den pool oluştur
        main_pool = self.get_best_main_pool(best_main_name, pool_size)
        plus_pool = self.get_best_plus_pool(best_plus_name, 5)
        
        # Son 50 çekilişte performans hesapla
        hits50 = []
        for i in range(min(50, len(self.df))):
            actual = set(self.get_main(self.df.iloc[-(i+1)]))
            hits50.append(len(set(main_pool) & actual))
        
        coverage50 = sum(1 for i in range(min(50, len(self.df)))
                         if set(self.get_main(self.df.iloc[-(i+1)])).issubset(set(main_pool)))
        
        random_hits = len(main_pool) * 5 / 34
        actual_hits = np.mean(hits50)
        improvement = (actual_hits - random_hits) / random_hits * 100 if random_hits > 0 else 0
        
        print("\n" + "=" * 70)
        print("🎯 ŞANS TOPU PATTERN MASTER V3 — RAPOR")
        print("   Süper Loto V2 mantığı ile yeniden yazıldı")
        print("=" * 70)
        print(f"📅 Son Çekiliş : {self.df['tarih'].max():%d.%m.%Y}")
        print(f"📊 Toplam      : {len(self.df)} çekiliş")
        
        print(f"\n{'─'*70}")
        print(f"🏆 ANA POOL ({len(main_pool)} sayı) ← En iyi pattern: {best_main_name}")
        print(f"{'─'*70}")
        print(f"  {main_pool}")
        print(f"\n  Ortalama isabet (son 50): {actual_hits:.2f}/{len(main_pool)} "
              f"(random={random_hits:.2f}, {improvement:+.1f}%)")
        print(f"  Coverage rate (son 50)  : {coverage50}/50 çekilişte 5/5 kapsandı "
              f"({coverage50/50:.1%})")
        
        print(f"\n{'─'*70}")
        print(f"🔥 ARTI POOL — En iyi pattern: {best_plus_name}")
        print(f"{'─'*70}")
        print(f"  {plus_pool}")
        
        print(f"\n{'─'*70}")
        print("📊 EN İYİ 5 ANA PATTERN")
        print(f"{'─'*70}")
        sorted_main = sorted(self.main_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_main[:5]):
            rastgele = 10 * (5 / 34)
            imp = (score - rastgele) / rastgele * 100
            print(f"  {i+1}. {name:40}: {score:.2f}/10 (%{imp:+.1f})")
        
        print(f"\n{'─'*70}")
        print("📊 EN İYİ 3 ARTI PATTERN")
        print(f"{'─'*70}")
        sorted_plus = sorted(self.plus_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_plus[:3]):
            rastgele = 4 * (1 / 14)
            imp = (score - rastgele) / rastgele * 100
            print(f"  {i+1}. {name:40}: {score:.2f}/4 (%{imp:+.1f})")
        
        print(f"\n{'─'*70}")
        print("⚠️  Eğlence amaçlıdır. Kazanç garantisi yoktur.")
        print("=" * 70)
        
        return {
            'main_pool': main_pool,
            'plus_pool': plus_pool,
            'best_main_pattern': best_main_name,
            'best_plus_pattern': best_plus_name,
            'avg_hits': float(actual_hits),
            'coverage50': int(coverage50),
            'improvement': float(improvement)
        }
    
    def save(self, result):
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/sanstopu_pattern_master_v3.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n💾 Kaydedildi: outputs/sanstopu_pattern_master_v3.json")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("🚀 ŞANS TOPU PATTERN MASTER V3")
    print("   Süper Loto V2 mantığı ile yeniden yazıldı")
    print("   YENİ ÖZELLİKLER:")
    print("     • Sayı Çiftleri (Pairwise) ve Üçlüler (Triplets)")
    print("     • Ortalama Aralığa Göre Gecikme (Overdue)")
    print("     • Onluk Blok (Gap) Analizi")
    print("     • Entropi (Karmaşıklık) Skoru ile Ağırlıklandırma")
    print("     • Kümeleme (Clustering) Analizi")
    print("=" * 70)

    bot = SansTopuPatternMasterV3()
    bot.load()
    bot.run_tests(test_size=100)
    result = bot.report(pool_size=17)  # Hedef 17 sayı (agresif)
    bot.save(result)
    print("\n✅ TAMAMLANDI!")


if __name__ == "__main__":
    main()
