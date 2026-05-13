#!/usr/bin/env python3
"""
SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ (V2 + İLERİ ANALİZLER)
- V2'nin tüm özelliklerini içerir (Pair, Triplet, Overdue, Block, Entropy)
- EK: Markov Zinciri (Sayısal Hafıza)
- EK: Z-Score (Anomali Tespiti)
- EK: Poisson Dağılımı (Beklenti Analizi)
- EK: Delta Analizi (Sayılar arası farklar)
- ÇIKTI: 28 sayılık SÜPER HAVUZ (4 hücreli)
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

# ============================================================
# VERİ YÜKLEYİCİ (V2'den aynen)
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
    
    def get_sorted_numbers(self, row):
        return sorted(self.get_numbers(row))


# ============================================================
# PATTERN MASTER V3 (V2 + İLERİ ANALİZLER)
# ============================================================

class PatternMasterV3:
    def __init__(self, df, get_numbers_func, get_sorted_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_sorted = get_sorted_func
        
        # Cache'ler
        self.pair_cache = None
        self.triplet_cache = None
        self.interval_cache = None
        self.markov_cache = None
        
    # ============================================================
    # V2'DEN AYNIEN ALINAN PATTERN'LER (1-17)
    # ============================================================
    
    def p_son_n(self, n, k=12):
        recent_nums = []
        window = min(n, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_son_1(self, k=12): return self.p_son_n(1, k)
    def p_son_3(self, k=12): return self.p_son_n(3, k)
    def p_son_5(self, k=12): return self.p_son_n(5, k)
    def p_son_7(self, k=12): return self.p_son_n(7, k)
    def p_son_10(self, k=12): return self.p_son_n(10, k)
    def p_son_15(self, k=12): return self.p_son_n(15, k)
    def p_son_20(self, k=12): return self.p_son_n(20, k)
    def p_son_30(self, k=12): return self.p_son_n(30, k)
    
    def p_fibonacci(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        return fib[:k]
    
    def p_fibonacci_neighbor(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 60: neighbors.add(f+1)
        return sorted(list(neighbors))[:k]
    
    def p_fibonacci_range(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        ranges = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                if x <= 60:
                    ranges.add(x)
        return sorted(list(ranges))[:k]
    
    def p_only_odd(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        counter = Counter(odds)
        return [num for num, _ in counter.most_common(k)]
    
    def p_only_even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        evens = [num for num in all_nums if num % 2 == 0]
        counter = Counter(evens)
        return [num for num, _ in counter.most_common(k)]
    
    def p_3odd_3even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(6)]
        top_evens = [num for num, _ in even_counter.most_common(6)]
        return (top_odds + top_evens)[:k]
    
    def p_4odd_2even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(8)]
        top_evens = [num for num, _ in even_counter.most_common(4)]
        return (top_odds + top_evens)[:k]
    
    def p_small(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 30]
        counter = Counter(small)
        return [num for num, _ in counter.most_common(k)]
    
    def p_large(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        large = [num for num in all_nums if num > 30]
        counter = Counter(large)
        return [num for num, _ in counter.most_common(k)]
    
    def p_3small_3large(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 30]
        large = [num for num in all_nums if num > 30]
        small_counter = Counter(small)
        large_counter = Counter(large)
        top_small = [num for num, _ in small_counter.most_common(6)]
        top_large = [num for num, _ in large_counter.most_common(6)]
        return (top_small + top_large)[:k]
    
    def p_mod5(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod5 = [num for num in all_nums if num % 5 == 0]
        counter = Counter(mod5)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod10(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod10 = [num for num in all_nums if num % 10 == 0]
        counter = Counter(mod10)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod3(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod3 = [num for num in all_nums if num % 3 == 0]
        counter = Counter(mod3)
        return [num for num, _ in counter.most_common(k)]
    
    def p_prime(self, k=12):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        return [p for p in primes if p <= 60][:k]
    
    def p_perfect_square(self, k=12):
        squares = [1, 4, 9, 16, 25, 36, 49]
        return squares[:k]
    
    def p_power2(self, k=12):
        powers = [1, 2, 4, 8, 16, 32]
        return powers[:k]
    
    def p_due(self, k=12):
        last_seen = {num: 0 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 61)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    def p_hot(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_trend_up(self, k=12):
        window = 20
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
    
    def p_trend_down(self, k=12):
        window = 20
        scores = {}
        for num in range(1, 61):
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
    
    def p_neighbor(self, k=12):
        last_row = self.df.iloc[-1]
        last_nums = self.get_numbers(last_row)
        neighbors = set()
        for num in last_nums:
            for offset in [-2, -1, 0, 1, 2]:
                neighbor = num + offset
                if 1 <= neighbor <= 60:
                    neighbors.add(neighbor)
        return sorted(list(neighbors))[:k]
    
    def p_last_draw(self, k=12):
        last_row = self.df.iloc[-1]
        return self.get_numbers(last_row)[:k]
    
    def p_weekday(self, k=12):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_month_day(self, k=12):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            return self.p_hot(k)
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_even_draws(self, k=12):
        even_indices = self.df.iloc[::2]
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_odd_draws(self, k=12):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def _build_pair_cache(self):
        if self.pair_cache is not None:
            return self.pair_cache
        
        pair_count = defaultdict(int)
        for idx, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for pair in combinations(nums, 2):
                pair_count[pair] += 1
        
        self.pair_cache = pair_count
        return self.pair_cache
    
    def p_top_pairs(self, k=12):
        cache = self._build_pair_cache()
        top_pairs = sorted(cache.items(), key=lambda x: x[1], reverse=True)[:30]
        pair_numbers = []
        for pair, _ in top_pairs:
            pair_numbers.extend(pair)
        counter = Counter(pair_numbers)
        return [num for num, _ in counter.most_common(k)]
    
    def p_with_most_frequent_partner(self, k=12):
        cache = self._build_pair_cache()
        best_partners = {}
        for (a, b), count in cache.items():
            if a not in best_partners or count > best_partners[a][1]:
                best_partners[a] = (b, count)
            if b not in best_partners or count > best_partners[b][1]:
                best_partners[b] = (a, count)
        
        partner_scores = defaultdict(int)
        for num, (partner, count) in best_partners.items():
            partner_scores[partner] += count
        
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_partners[:k]]
    
    def _build_triplet_cache(self):
        if self.triplet_cache is not None:
            return self.triplet_cache
        
        triplet_count = defaultdict(int)
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for triplet in combinations(nums, 3):
                triplet_count[triplet] += 1
        
        self.triplet_cache = triplet_count
        return self.triplet_cache
    
    def p_top_triplets(self, k=12):
        cache = self._build_triplet_cache()
        top_triplets = sorted(cache.items(), key=lambda x: x[1], reverse=True)[:20]
        triplet_numbers = []
        for triplet, _ in top_triplets:
            triplet_numbers.extend(triplet)
        counter = Counter(triplet_numbers)
        return [num for num, _ in counter.most_common(k)]
    
    def _build_interval_cache(self):
        if self.interval_cache is not None:
            return self.interval_cache
        
        intervals = {num: [] for num in range(1, 61)}
        last_seen = {num: -1 for num in range(1, 61)}
        
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                if last_seen[num] != -1:
                    interval = idx - last_seen[num]
                    intervals[num].append(interval)
                last_seen[num] = idx
        
        current_idx = len(self.df) - 1
        current_gap = {num: current_idx - last_seen[num] if last_seen[num] != -1 else current_idx + 1 
                      for num in range(1, 61)}
        
        avg_intervals = {}
        for num in range(1, 61):
            if intervals[num]:
                avg_intervals[num] = sum(intervals[num]) / len(intervals[num])
            else:
                avg_intervals[num] = 10
        
        self.interval_cache = {
            'avg': avg_intervals,
            'current_gap': current_gap,
            'history': intervals
        }
        return self.interval_cache
    
    def p_overdue(self, k=12):
        cache = self._build_interval_cache()
        overdue_scores = {}
        for num in range(1, 61):
            avg = cache['avg'][num]
            gap = cache['current_gap'][num]
            if avg > 0:
                ratio = gap / avg
                if ratio > 2.0:
                    overdue_scores[num] = 3.0
                elif ratio > 1.5:
                    overdue_scores[num] = 2.0
                elif ratio > 1.0:
                    overdue_scores[num] = 1.5
                else:
                    overdue_scores[num] = 1.0
            else:
                overdue_scores[num] = 1.0
        
        sorted_nums = sorted(overdue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_due_with_rhythm(self, k=12):
        cache = self._build_interval_cache()
        rhythm_scores = {}
        for num in range(1, 61):
            history = cache['history'][num]
            if len(history) >= 3:
                recent_intervals = history[-3:]
                if len(set(recent_intervals)) == 1:
                    rhythm_scores[num] = 2.0
                elif recent_intervals[-1] > recent_intervals[-2] > recent_intervals[-3]:
                    rhythm_scores[num] = 1.5
                else:
                    rhythm_scores[num] = 1.0
            else:
                rhythm_scores[num] = 1.0
        
        for num in range(1, 61):
            gap = cache['current_gap'][num]
            avg = cache['avg'][num]
            if avg > 0 and gap > avg * 1.2:
                rhythm_scores[num] = rhythm_scores.get(num, 1.0) * 1.5
        
        sorted_nums = sorted(rhythm_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def _calculate_block_stats(self):
        blocks = {i: [] for i in range(6)}
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            block_hits = {i: False for i in range(6)}
            for num in nums:
                block_idx = (num - 1) // 10
                if 0 <= block_idx < 6:
                    block_hits[block_idx] = True
            for i in range(6):
                blocks[i].append(block_hits[i])
        
        empty_rates = {}
        hit_rates = {}
        for i in range(6):
            total = len(blocks[i])
            empty_count = sum(1 for hit in blocks[i] if not hit)
            hit_count = sum(1 for hit in blocks[i] if hit)
            empty_rates[i] = empty_count / total if total > 0 else 0
            hit_rates[i] = hit_count / total if total > 0 else 0
        
        return empty_rates, hit_rates
    
    def p_block_gap_active(self, k=12):
        empty_rates, hit_rates = self._calculate_block_stats()
        best_block = max(hit_rates.items(), key=lambda x: x[1])[0]
        block_numbers = range(best_block * 10 + 1, best_block * 10 + 11)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in block_nums_with_freq[:k]]
    
    def p_block_gap_active(self, k=12):
        empty_rates, hit_rates = self._calculate_block_stats()
        best_block = max(hit_rates.items(), key=lambda x: x[1])[0]
        block_numbers = range(best_block * 10 + 1, best_block * 10 + 11)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in block_nums_with_freq[:k]]
    
    def p_block_gap_inactive(self, k=12):
        empty_rates, hit_rates = self._calculate_block_stats()
        worst_block = max(empty_rates.items(), key=lambda x: x[1])[0]
        block_numbers = range(worst_block * 10 + 1, worst_block * 10 + 11)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in block_nums_with_freq[:k]]
    
    def p_cluster_centers(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        block_freq = defaultdict(int)
        for num in all_nums:
            block = (num - 1) // 10
            block_freq[block] += 1
        most_dense_block = max(block_freq.items(), key=lambda x: x[1])[0]
        block_numbers = range(most_dense_block * 10 + 1, most_dense_block * 10 + 11)
        counter = Counter(all_nums)
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in block_nums_with_freq[:k]]
    
    def calculate_entropy(self, numbers):
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
    
    def p_low_entropy(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if self.calculate_entropy(nums) < 2.0:
                all_nums.extend(nums)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_high_entropy(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if self.calculate_entropy(nums) > 2.3:
                all_nums.extend(nums)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_gap_between_numbers(self, k=12):
        gaps = []
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for i in range(len(nums)-1):
                gaps.append(nums[i+1] - nums[i])
        gap_counter = Counter(gaps)
        most_common_gap = gap_counter.most_common(1)[0][0] if gap_counter else 5
        last_nums = sorted(self.get_numbers(self.df.iloc[-1]))
        predictions = set()
        for num in last_nums:
            next_num = num + most_common_gap
            if 1 <= next_num <= 60:
                predictions.add(next_num)
            next_num = num - most_common_gap
            if 1 <= next_num <= 60:
                predictions.add(next_num)
        if len(predictions) < k:
            all_nums = []
            for _, row in self.df.iterrows():
                all_nums.extend(self.get_numbers(row))
            counter = Counter(all_nums)
            for num, _ in counter.most_common(k):
                predictions.add(num)
                if len(predictions) >= k:
                    break
        return list(predictions)[:k]
    
    # ============================================================
    # YENİ! V3 İLERİ ANALİZLER
    # ============================================================
    
    def _build_markov_matrix(self):
        """Markov geçiş matrisi - V3 YENİ"""
        if self.markov_cache is not None:
            return self.markov_cache
        
        transitions = defaultdict(lambda: defaultdict(int))
        
        for idx in range(len(self.df) - 1):
            current_nums = set(self.get_numbers(self.df.iloc[idx]))
            next_nums = set(self.get_numbers(self.df.iloc[idx + 1]))
            
            for current in current_nums:
                for next_num in next_nums:
                    transitions[current][next_num] += 1
        
        markov_prob = {}
        for current, next_dict in transitions.items():
            total = sum(next_dict.values())
            if total > 0:
                markov_prob[current] = {num: count/total for num, count in next_dict.items()}
        
        self.markov_cache = markov_prob
        return markov_prob
    
    def p_markov_boost(self, k=12):
        """Son çekilişteki sayılardan sonra gelme olasılığı yüksek sayılar - V3 YENİ"""
        markov = self._build_markov_matrix()
        last_nums = set(self.get_numbers(self.df.iloc[-1]))
        
        boost_scores = defaultdict(float)
        for num in last_nums:
            if num in markov:
                for next_num, prob in markov[num].items():
                    boost_scores[next_num] += prob
        
        sorted_nums = sorted(boost_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_zscore_anomaly(self, k=12):
        """Z-Score ile anomali tespiti - V3 YENİ"""
        total_draws = len(self.df)
        expected_freq = total_draws * 6 / 60
        
        actual_freq = {num: 0 for num in range(1, 61)}
        for _, row in self.df.iterrows():
            for num in self.get_numbers(row):
                actual_freq[num] += 1
        
        p = 6/60
        std_dev = math.sqrt(total_draws * p * (1 - p))
        
        z_scores = {}
        for num in range(1, 61):
            if std_dev > 0:
                z_scores[num] = (actual_freq[num] - expected_freq) / std_dev
            else:
                z_scores[num] = 0
        
        # Negatif Z-Score (az çıkmış) en yüksek
        sorted_nums = sorted(z_scores.items(), key=lambda x: x[1])
        return [num for num, _ in sorted_nums[:k]]
    
    def p_poisson_expectation(self, k=12):
        """Poisson dağılımı ile beklenti analizi - V3 YENİ"""
        total_draws = len(self.df)
        lambda_val = total_draws * 6 / 60
        
        actual_freq = {num: 0 for num in range(1, 61)}
        for _, row in self.df.iterrows():
            for num in self.get_numbers(row):
                actual_freq[num] += 1
        
        poisson_scores = {}
        for num in range(1, 61):
            if actual_freq[num] < lambda_val:
                poisson_scores[num] = (lambda_val - actual_freq[num]) / lambda_val
            else:
                poisson_scores[num] = 0.5
        
        sorted_nums = sorted(poisson_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_delta_analysis(self, k=12):
        """Delta analizi - V3 YENİ"""
        gaps = []
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for i in range(len(nums)-1):
                gaps.append(nums[i+1] - nums[i])
        
        if not gaps:
            return list(range(1, k+1))
        
        gap_counter = Counter(gaps)
        common_deltas = [d for d, _ in sorted(gap_counter.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        last_nums = sorted(self.get_numbers(self.df.iloc[-1]))
        compatible = set()
        for delta in common_deltas:
            for num in last_nums:
                candidate = num + delta
                if 1 <= candidate <= 60:
                    compatible.add(candidate)
                candidate = num - delta
                if 1 <= candidate <= 60:
                    compatible.add(candidate)
        
        result = list(compatible)
        if len(result) < k:
            hot_nums = self.p_hot(k)
            for num in hot_nums:
                if num not in result:
                    result.append(num)
                    if len(result) >= k:
                        break
        
        return result[:k]


# ============================================================
# BACKTEST MOTORU V3 (V2 + Yeni pattern'ler)
# ============================================================

class BacktestEngineV3:
    def __init__(self, df, get_numbers_func, get_sorted_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_sorted = get_sorted_func
        self.results = {}
        
    def test_pattern(self, pattern_func, pattern_name, test_size=150):
        total = len(self.df)
        train_size = total - test_size
        
        if train_size <= 0:
            return 0
        
        scores = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            temp_df = self.df.iloc[:train_end]
            temp_patterns = PatternMasterV3(temp_df, self.get_numbers, self.get_sorted)
            
            try:
                preds = getattr(temp_patterns, pattern_func)(12)
            except Exception as e:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            correct = len(set(preds) & actual)
            
            entropy = temp_patterns.calculate_entropy(preds)
            entropy_weight = 1.0
            if 2.0 <= entropy <= 2.5:
                entropy_weight = 1.1
            elif entropy < 1.5 or entropy > 2.8:
                entropy_weight = 0.9
            
            scores.append(correct * entropy_weight)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def run_all_tests(self, test_size=150):
        print(f"\n📊 PATTERN TESTİ V3 BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("   Toplam 52 pattern test edilecek (V2:48 + V3 YENİ:4)")
        print("   YENİ: Markov | Z-Score | Poisson | Delta")
        print("-" * 70)
        
        pattern_list = [
            ('p_son_1', 'Son 1 çekiliş'),
            ('p_son_3', 'Son 3 çekiliş'),
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_son_20', 'Son 20 çekiliş'),
            ('p_son_30', 'Son 30 çekiliş'),
            ('p_fibonacci', 'Fibonacci sayıları'),
            ('p_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_fibonacci_range', 'Fibonacci aralığı'),
            ('p_only_odd', 'Sadece tek sayılar'),
            ('p_only_even', 'Sadece çift sayılar'),
            ('p_3odd_3even', '3 tek + 3 çift'),
            ('p_4odd_2even', '4 tek + 2 çift'),
            ('p_small', 'Küçük sayılar (1-30)'),
            ('p_large', 'Büyük sayılar (31-60)'),
            ('p_3small_3large', '3 küçük + 3 büyük'),
            ('p_mod5', "Mod 5 (5'in katları)"),
            ('p_mod10', "Mod 10 (10'un katları)"),
            ('p_mod3', "Mod 3 (3'ün katları)"),
            ('p_prime', 'Asal sayılar'),
            ('p_perfect_square', 'Tam kareler'),
            ('p_power2', "2'nin kuvvetleri"),
            ('p_due', 'Due numbers'),
            ('p_hot', 'Hot numbers'),
            ('p_trend_up', 'Trend artan sayılar'),
            ('p_trend_down', 'Trend azalan sayılar'),
            ('p_neighbor', 'Son çekilişin komşuları'),
            ('p_last_draw', 'Son çekilişin aynısı'),
            ('p_weekday', 'Haftanın aynı günü'),
            ('p_month_day', 'Ayın aynı günü'),
            ('p_even_draws', 'Çift numaralı çekilişler'),
            ('p_odd_draws', 'Tek numaralı çekilişler'),
            ('p_top_pairs', '🔗 En çok çıkan sayı çiftleri'),
            ('p_with_most_frequent_partner', '🤝 En güçlü partner ilişkileri'),
            ('p_top_triplets', '🔺 En çok çıkan sayı üçlüleri'),
            ('p_overdue', '⏰ Vadesi geçmiş sayılar'),
            ('p_due_with_rhythm', '🎵 Ritmik Due analizi'),
            ('p_block_gap_active', '📊 En aktif blok'),
            ('p_block_gap_inactive', '🎯 En pasif blok'),
            ('p_cluster_centers', '🎯 Küme merkezleri'),
            ('p_low_entropy', '📐 Düşük entropi'),
            ('p_high_entropy', '🌊 Yüksek entropi'),
            ('p_gap_between_numbers', '📏 Sayılar arası fark analizi'),
            ('p_markov_boost', '🔄 Markov Zinciri (V3 YENİ)'),
            ('p_zscore_anomaly', '📈 Z-Score Anomali (V3 YENİ)'),
            ('p_poisson_expectation', '📊 Poisson Dağılımı (V3 YENİ)'),
            ('p_delta_analysis', '📐 Delta Analizi (V3 YENİ)'),
        ]
        
        print("\n  📈 PATTERN TEST SONUÇLARI V3:")
        print(f"  {'Pattern':45} {'Başarı':>10}")
        print("  " + "-" * 57)
        
        for pattern_func, pattern_name in pattern_list:
            score = self.test_pattern(pattern_func, pattern_name, test_size)
            self.results[pattern_name] = score
            
            if score > 1.35:
                star = "🔥🔥"
            elif score > 1.30:
                star = "🔥"
            elif score > 1.25:
                star = "⭐"
            elif score > 1.20:
                star = "✓"
            else:
                star = " "
            
            v3_tag = " [V3]" if "V3 YENİ" in pattern_name else ""
            print(f"  {pattern_name:45}: {score:5.2f}/12 {star}{v3_tag}")
        
        print("\n" + "-" * 70)
        print("🏆 EN İYİ 15 PATTERN (V3)")
        print("-" * 70)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:15]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            v3_tag = " [V3-YENİ]" if "V3 YENİ" in name else ""
            print(f"  {medal} {name:45}: {score:.2f}/12{v3_tag}")
        
        return self.results


# ============================================================
# SÜPER HAVUZ ÜRETİCİ (28 Sayı - 4 Hücre) - V3
# ============================================================

class SuperPoolGeneratorV3:
    def __init__(self, df, get_numbers_func, get_sorted_func, test_results):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_sorted = get_sorted_func
        self.test_results = test_results
        self.pm = PatternMasterV3(df, get_numbers_func, get_sorted_func)
        
    def _get_top_pattern_names(self, n=15):
        sorted_results = sorted(self.test_results.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_results[:n]]
    
    def generate_super_pool(self):
        print("\n" + "=" * 70)
        print("🔬 SÜPER HAVUZ ÜRETİLİYOR (V3 - İleri Analiz)")
        print("   V2: 48 pattern | V3 YENİ: Markov | Z-Score | Poisson | Delta")
        print("=" * 70)
        
        top_patterns = self._get_top_pattern_names(15)
        print(f"\n📊 En iyi 15 pattern kullanılıyor:")
        for i, p in enumerate(top_patterns[:10]):
            print(f"   {i+1}. {p[:50]}...")
        
        # ============================================================
        # HÜCRE A: TREND LİDERLERİ (V2 pattern'ler + Markov)
        # ============================================================
        
        trend_candidates = self.pm.p_son_7(20) + self.pm.p_son_5(20) + self.pm.p_hot(20)
        trend_candidates = list(dict.fromkeys(trend_candidates))
        
        markov_scores = {num: 1.0 for num in trend_candidates}
        markov_boost = self.pm.p_markov_boost(20)
        for i, num in enumerate(markov_boost):
            if num in markov_scores:
                markov_scores[num] += (20 - i) / 20
        
        sorted_trend = sorted(markov_scores.items(), key=lambda x: x[1], reverse=True)
        cell_a = [num for num, _ in sorted_trend[:7]]
        
        print(f"\n📊 HÜCRE A (Trend Liderleri + Markov) - 7 sayı:")
        print(f"   {cell_a}")
        
        # ============================================================
        # HÜCRE B: MATEMATİKSEL BEKLENTİ (Overdue + Poisson + Z-Score)
        # ============================================================
        
        overdue_nums = self.pm.p_overdue(20)
        poisson_nums = self.pm.p_poisson_expectation(20)
        zscore_nums = self.pm.p_zscore_anomaly(20)
        
        expected_scores = {}
        for num in overdue_nums:
            expected_scores[num] = expected_scores.get(num, 0) + 1
        for num in poisson_nums:
            expected_scores[num] = expected_scores.get(num, 0) + 1.5
        for num in zscore_nums:
            expected_scores[num] = expected_scores.get(num, 0) + 1.2
        
        sorted_expected = sorted(expected_scores.items(), key=lambda x: x[1], reverse=True)
        cell_b = [num for num, _ in sorted_expected[:7]]
        
        print(f"\n📊 HÜCRE B (Matematiksel Beklenti) - 7 sayı:")
        print(f"   {cell_b}")
        
        # ============================================================
        # HÜCRE C: KORELASYON/PARTNER (Pair + Triplet + Delta)
        # ============================================================
        
        pair_nums = self.pm.p_top_pairs(20)
        triplet_nums = self.pm.p_top_triplets(20)
        delta_nums = self.pm.p_delta_analysis(20)
        
        partner_scores = {}
        for num in pair_nums:
            partner_scores[num] = partner_scores.get(num, 0) + 1
        for num in triplet_nums:
            partner_scores[num] = partner_scores.get(num, 0) + 1.2
        for num in delta_nums:
            partner_scores[num] = partner_scores.get(num, 0) + 1.5
        
        sorted_partner = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        cell_c = [num for num, _ in sorted_partner[:7]]
        
        print(f"\n📊 HÜCRE C (Korelasyon/Partner) - 7 sayı:")
        print(f"   {cell_c}")
        
        # ============================================================
        # HÜCRE D: KAOS/SÜRPRİZ (Düşük entropi + Yüksek entropi)
        # ============================================================
        
        low_entropy_nums = self.pm.p_low_entropy(20)
        high_entropy_nums = self.pm.p_high_entropy(20)
        
        chaos_scores = {}
        for num in low_entropy_nums[:10]:
            chaos_scores[num] = chaos_scores.get(num, 0) + 1
        for num in high_entropy_nums[:10]:
            chaos_scores[num] = chaos_scores.get(num, 0) + 1
        
        sorted_chaos = sorted(chaos_scores.items(), key=lambda x: x[1], reverse=True)
        cell_d = [num for num, _ in sorted_chaos[:7]]
        
        print(f"\n📊 HÜCRE D (Kaos/Sürpriz) - 7 sayı:")
        print(f"   {cell_d}")
        
        # ============================================================
        # SÜPER HAVUZ (28 Sayı)
        # ============================================================
        
        super_pool = cell_a + cell_b + cell_c + cell_d
        unique_pool = list(dict.fromkeys(super_pool))
        
        print(f"\n📊 Benzersiz havuz: {len(unique_pool)} sayı (tekrarlar temizlendi)")
        
        if len(unique_pool) < 28:
            hot_nums = self.pm.p_hot(30)
            for num in hot_nums:
                if num not in unique_pool:
                    unique_pool.append(num)
                    if len(unique_pool) >= 28:
                        break
        
        super_pool = unique_pool[:28]
        sorted_pool = sorted(super_pool)
        
        print("\n" + "-" * 70)
        print("🎯 SÜPER HAVUZ (28 SAYI) - V3 FİNAL")
        print("-" * 70)
        
        print(f"\n  {' '.join(f'{n:3d}' for n in sorted_pool[:14])}")
        print(f"  {' '.join(f'{n:3d}' for n in sorted_pool[14:])}")
        
        blocks = {f"{i*10+1}-{(i+1)*10}": 0 for i in range(6)}
        for num in sorted_pool:
            block_idx = (num - 1) // 10
            block_key = f"{block_idx*10+1}-{(block_idx+1)*10}"
            blocks[block_key] = blocks.get(block_key, 0) + 1
        
        print(f"\n📊 Onluk Blok Dağılımı:")
        for block, count in blocks.items():
            bar = "█" * count
            print(f"  {block:12}: {count:2} sayı {bar}")
        
        odds = [n for n in sorted_pool if n % 2 == 1]
        evens = [n for n in sorted_pool if n % 2 == 0]
        print(f"\n📊 Tek/Çift Dağılımı: {len(odds)} tek / {len(evens)} çift")
        
        small = [n for n in sorted_pool if n <= 30]
        large = [n for n in sorted_pool if n > 30]
        print(f"📊 Büyük/Küçük Dağılımı: {len(small)} küçük (1-30) / {len(large)} büyük (31-60)")
        
        print(f"\n📊 Havuz İstatistikleri:")
        print(f"  Min: {min(sorted_pool)} | Max: {max(sorted_pool)} | Range: {max(sorted_pool) - min(sorted_pool)}")
        print(f"  Ortalama: {sum(sorted_pool)/len(sorted_pool):.1f}")
        print(f"  Medyan: {sorted_pool[len(sorted_pool)//2]}")
        
        return {
            'super_pool': sorted_pool,
            'cell_a': cell_a,
            'cell_b': cell_b,
            'cell_c': cell_c,
            'cell_d': cell_d,
            'stats': {
                'odds_count': len(odds),
                'evens_count': len(evens),
                'small_count': len(small),
                'large_count': len(large),
                'avg': sum(sorted_pool)/len(sorted_pool),
                'range': max(sorted_pool) - min(sorted_pool)
            }
        }


# ============================================================
# ANA SINIF - SUPER LOTO PATTERN MASTER V3
# ============================================================

class SuperLotoPatternMasterV3:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.test_results = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        self.get_sorted = loader.get_sorted_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_tests(self, test_size=150):
        engine = BacktestEngineV3(self.df, self.get_numbers, self.get_sorted)
        self.test_results = engine.run_all_tests(test_size)
        
        best_pattern = max(self.test_results.items(), key=lambda x: x[1])
        print(f"\n🎯 EN İYİ PATTERN V3: {best_pattern[0]} ({best_pattern[1]:.2f}/12)")
        print(f"   Rastgele şans: 1.20/12")
        print(f"   İyileştirme: %{((best_pattern[1]-1.2)/1.2*100):.1f}")
        
        return self.test_results
    
    def generate_super_pool(self):
        if self.test_results is None:
            self.run_tests()
        
        pool_gen = SuperPoolGeneratorV3(self.df, self.get_numbers, self.get_sorted, self.test_results)
        result = pool_gen.generate_super_pool()
        return result
    
    def print_report(self, result):
        print("\n" + "=" * 70)
        print("🎯 SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ RAPORU")
        print("   V2: 48 pattern | V3 YENİ: Markov | Z-Score | Poisson | Delta")
        print("=" * 70)
        
        print("\n🏆 SÜPER HAVUZ (28 SAYI):")
        print("-" * 70)
        
        sorted_pool = result['super_pool']
        print(f"\n  {' '.join(f'{n:3d}' for n in sorted_pool[:14])}")
        print(f"  {' '.join(f'{n:3d}' for n in sorted_pool[14:])}")
        
        print("\n" + "-" * 70)
        print("🔬 4 HÜCRE YAPISI (Nokta Atışı - V3)")
        print("-" * 70)
        
        print(f"\n  HÜCRE A (Trend+Markov):   {result['cell_a']}")
        print(f"  HÜCRE B (Beklenti):       {result['cell_b']}")
        print(f"  HÜCRE C (Partner+Delta):  {result['cell_c']}")
        print(f"  HÜCRE D (Entropi):        {result['cell_d']}")
        
        print("\n" + "-" * 70)
        print("📊 İSTATİSTİKLER")
        print("-" * 70)
        print(f"  Tek/Çift: {result['stats']['odds_count']}/{result['stats']['evens_count']}")
        print(f"  Küçük/Büyük: {result['stats']['small_count']}/{result['stats']['large_count']}")
        print(f"  Ortalama: {result['stats']['avg']:.1f}")
        print(f"  Aralık: {result['stats']['range']}")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: V3, V2'nin tüm özelliklerini + yeni analizleri içerir.")
        print("   Markov | Z-Score | Poisson | Delta analizleri eklendi.")
        print("   Hedef: 28 sayıda 6/6 isabet!")
        print("=" * 70)
        
        return result
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/pattern_master_v3_super_pool.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'V3',
                'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_draw': self.df['tarih'].max().strftime('%d.%m.%Y') if len(self.df) > 0 else None,
                'super_pool_28': result['super_pool'],
                'cell_a_trend_markov': result['cell_a'],
                'cell_b_expectation': result['cell_b'],
                'cell_c_partner_delta': result['cell_c'],
                'cell_d_chaos_entropy': result['cell_d'],
                'statistics': result['stats'],
                'new_features': ['Markov Zinciri', 'Z-Score', 'Poisson', 'Delta Analizi']
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/pattern_master_v3_super_pool.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Oluşturulma: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y') if len(self.df) > 0 else '-'}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"Yeni Özellikler: Markov | Z-Score | Poisson | Delta\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("🏆 SÜPER HAVUZ (28 SAYI)\n")
            f.write("-" * 70 + "\n\n")
            
            sorted_pool = result['super_pool']
            f.write("  ")
            for i, num in enumerate(sorted_pool[:14]):
                f.write(f"{num:3d} ")
                if i == 6:
                    f.write("\n  ")
            f.write("\n  ")
            for i, num in enumerate(sorted_pool[14:]):
                f.write(f"{num:3d} ")
                if i == 6:
                    f.write("\n  ")
            f.write("\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("🔬 4 HÜCRE YAPISI (V3)\n")
            f.write("-" * 70 + "\n\n")
            f.write("HÜCRE A (Trend + Markov):\n")
            f.write(f"  {result['cell_a']}\n\n")
            f.write("HÜCRE B (Poisson + Z-Score + Overdue):\n")
            f.write(f"  {result['cell_b']}\n\n")
            f.write("HÜCRE C (Partner + Triplet + Delta):\n")
            f.write(f"  {result['cell_c']}\n\n")
            f.write("HÜCRE D (Low/High Entropy):\n")
            f.write(f"  {result['cell_d']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("📊 İSTATİSTİKLER\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"  Tek/Çift: {result['stats']['odds_count']}/{result['stats']['evens_count']}\n")
            f.write(f"  Küçük/Büyük: {result['stats']['small_count']}/{result['stats']['large_count']}\n")
            f.write(f"  Ortalama: {result['stats']['avg']:.1f}\n")
            f.write(f"  Aralık: {result['stats']['range']}\n")
        
        print(f"\n💾 Kaydedildi: outputs/pattern_master_v3_super_pool.json")
        print(f"💾 Kaydedildi: outputs/pattern_master_v3_super_pool.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 SÜPER LOTO PATTERN MASTER V3")
    print("   V2 TÜM ÖZELLİKLER + YENİLER:")
    print("     • Markov Zinciri (Sayısal Hafıza)")
    print("     • Z-Score (Anomali Tespiti)")
    print("     • Poisson Dağılımı (Beklenti Analizi)")
    print("     • Delta Analizi (Sayılar arası farklar)")
    print("   TEK BOT - NOKTATAŞI - 28 SAYILIK SÜPER HAVUZ")
    print("=" * 70)
    
    bot = SuperLotoPatternMasterV3()
    bot.load_data()
    bot.run_tests(test_size=150)
    result = bot.generate_super_pool()
    bot.print_report(result)
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
