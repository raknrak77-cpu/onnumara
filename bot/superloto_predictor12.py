#!/usr/bin/env python3
"""
SÜPER LOTO 6/12 BOTU
60 sayı içinden en güçlü 12 sayıyı önerir
Çıktı: 12 sayı (sen içinden 6 seçersin)
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
# MODELLER (12 sayı için)
# ============================================================

class Models:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    def recent(self, n=12, window=5):
        recent_nums = []
        window = min(window, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        result = [num for num, _ in counter.most_common(n)]
        if len(result) < n:
            freq = self.frequency(n*2)
            for num in freq:
                if num not in result:
                    result.append(num)
                if len(result) == n:
                    break
        return result[:n]
    
    def frequency(self, n=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    def due(self, n=12):
        last_seen = {num: 0 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 61)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    def weighted(self, n=12, decay=0.95):
        weighted_counts = {}
        for idx, row in self.df.iterrows():
            weight = decay ** (len(self.df) - idx - 1)
            for num in self.get_numbers(row):
                weighted_counts[num] = weighted_counts.get(num, 0) + weight
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    def trend(self, n=12, window=20):
        scores = {}
        for num in range(1, 61):
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


# ============================================================
# BACKTEST (12 sayı için)
# ============================================================

class Backtest:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.results = {}
        
    def test_model(self, model_func, model_name, test_size=50, **kwargs):
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
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            
            temp_models = Models(train_df, self.get_numbers)
            
            if model_name == 'recent':
                window = kwargs.get('window', 5)
                preds = temp_models.recent(12, window)
            elif model_name == 'frequency':
                preds = temp_models.frequency(12)
            elif model_name == 'due':
                preds = temp_models.due(12)
            elif model_name == 'weighted':
                decay = kwargs.get('decay', 0.95)
                preds = temp_models.weighted(12, decay)
            elif model_name == 'trend':
                window = kwargs.get('window', 20)
                preds = temp_models.trend(12, window)
            else:
                preds = []
            
            correct = len(set(preds) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def find_best_window_for_recent(self, test_size=50):
        print("  🔍 Son N çekiliş modeli optimize ediliyor...")
        best_score = 0
        best_window = 5
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        
        for window in windows:
            if window > len(self.df) - test_size:
                continue
            score = self.test_model(None, 'recent', test_size, window=window)
            print(f"     window={window:2d} -> {score:.2f}/12 doğru")
            if score > best_score:
                best_score = score
                best_window = window
        
        print(f"  ✅ En iyi pencere: {best_window} (ortalama {best_score:.2f}/12)")
        return best_window, best_score
    
    def find_best_decay_for_weighted(self, test_size=50):
        print("  🔍 Ağırlıklı frekans modeli optimize ediliyor...")
        best_score = 0
        best_decay = 0.95
        decays = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
        
        for decay in decays:
            score = self.test_model(None, 'weighted', test_size, decay=decay)
            print(f"     decay={decay} -> {score:.2f}/12 doğru")
            if score > best_score:
                best_score = score
                best_decay = decay
        
        print(f"  ✅ En iyi decay: {best_decay} (ortalama {best_score:.2f}/12)")
        return best_decay, best_score
    
    def run_all_backtests(self, test_size=50):
        print(f"\n📊 BACKTEST BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("-" * 50)
        
        print("\n  📈 TEMEL MODELLER:")
        models_to_test = ['frequency', 'due']
        for model in models_to_test:
            score = self.test_model(None, model, test_size)
            self.results[model] = score
            print(f"     {model}: {score:.2f}/12 doğru")
        
        best_window, recent_score = self.find_best_window_for_recent(test_size)
        self.results['recent'] = recent_score
        self.results['recent_best_window'] = best_window
        
        best_decay, weighted_score = self.find_best_decay_for_weighted(test_size)
        self.results['weighted'] = weighted_score
        self.results['weighted_best_decay'] = best_decay
        
        trend_score = self.test_model(None, 'trend', test_size)
        self.results['trend'] = trend_score
        
        print("\n" + "-" * 50)
        print("🏆 MODEL BAŞARI SIRALAMASI (12'de kaç doğru):")
        print("-" * 50)
        
        sorted_results = sorted([(k, v) for k, v in self.results.items() if isinstance(v, (int, float))], key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_results):
            stars = "⭐" * int(score // 1) + "☆" * (6 - int(score // 1))
            print(f"  {i+1}. {model:12}: {score:.2f}/12 {stars}")
        
        return self.results


# ============================================================
# ANA TAHMİN SINIFI (12 sayı)
# ============================================================

class SuperLoto12Predictor:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.backtest_results = None
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
        self.backtest_results = bt.run_all_backtests(test_size)
        
        self.optimal_settings = {
            'best_model': max([(k, v) for k, v in self.backtest_results.items() if isinstance(v, (int, float))], key=lambda x: x[1])[0],
            'recent_window': self.backtest_results.get('recent_best_window', 5),
            'weighted_decay': self.backtest_results.get('weighted_best_decay', 0.95)
        }
        
        print(f"\n🎯 EN İYİ MODEL: {self.optimal_settings['best_model']}")
        return self.backtest_results
    
    def get_optimized_12(self):
        recent_window = self.optimal_settings.get('recent_window', 5)
        weighted_decay = self.optimal_settings.get('weighted_decay', 0.95)
        
        temp_models = Models(self.df, self.get_numbers)
        
        models_pred = {
            'recent': temp_models.recent(20, recent_window),
            'frequency': temp_models.frequency(20),
            'due': temp_models.due(20),
            'weighted': temp_models.weighted(20, weighted_decay),
            'trend': temp_models.trend(20)
        }
        
        weights = {
            'recent': self.backtest_results.get('recent', 1.0) if self.backtest_results else 1.0,
            'frequency': self.backtest_results.get('frequency', 1.0) if self.backtest_results else 1.0,
            'due': self.backtest_results.get('due', 1.0) if self.backtest_results else 1.0,
            'weighted': self.backtest_results.get('weighted', 1.0) if self.backtest_results else 1.0,
            'trend': self.backtest_results.get('trend', 1.0) if self.backtest_results else 1.0
        }
        
        all_votes = []
        for model_name, preds in models_pred.items():
            weight = weights.get(model_name, 1.0)
            for rank, num in enumerate(preds):
                vote_count = int((20 - rank) * weight)
                all_votes.extend([num] * max(1, vote_count))
        
        counter = Counter(all_votes)
        final_12 = [num for num, _ in counter.most_common(12)]
        
        return final_12
    
    def print_report(self):
        if self.backtest_results is None:
            self.run_backtest()
        
        final_12 = self.get_optimized_12()
        
        # Bu 12 sayı içinde en sık çıkan 6 sayıyı bul
        freq_in_12 = {}
        for num in final_12:
            count = 0
            for _, row in self.df.iterrows():
                if num in self.get_numbers(row):
                    count += 1
            freq_in_12[num] = count
        
        sorted_by_freq = sorted(freq_in_12.items(), key=lambda x: x[1], reverse=True)
        best_6 = [num for num, _ in sorted_by_freq[:6]]
        
        print("\n" + "=" * 60)
        print("🎯 SÜPER LOTO 6/12 BOTU")
        print("   60 sayı içinden en güçlü 12 sayı")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🎯 En İyi Model: {self.optimal_settings.get('best_model', 'unknown')}")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 12 SAYI")
        print("-" * 60)
        
        for i in range(0, len(final_12), 4):
            group = final_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 ÖNERİLEN 6 SAYI (Bu 12'nin içinden en güçlüler)")
        print("-" * 60)
        print(f"\n  🌟🌟🌟  {best_6}  🌟🌟🌟")
        
        print("\n" + "-" * 60)
        print("📊 BACKTEST SONUÇLARI")
        print("-" * 60)
        
        for model, score in sorted(self.backtest_results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
            if isinstance(score, (int, float)):
                print(f"  {model:12}: {score:.2f}/12")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu 12 sayı içinden 6 tane seçip oynayabilirsiniz.")
        print("   Kombinasyon sayısı: C(12,6) = 924")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return {'final_12': final_12, 'best_6': best_6}
    
    def save_results(self, final_12, best_6, filename="superloto_12_predictions.json"):
        os.makedirs('outputs', exist_ok=True)
        
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump({
                'best_12_numbers': final_12,
                'recommended_6_numbers': best_6,
                'backtest_results': self.backtest_results,
                'optimal_settings': self.optimal_settings
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/superloto_12_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 SÜPER LOTO 6/12 TAHMİN RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"En İyi Model: {self.optimal_settings.get('best_model', 'unknown')}\n\n")
            f.write("EN GÜÇLÜ 12 SAYI:\n")
            f.write(str(final_12) + "\n\n")
            f.write("🎯 ÖNERİLEN 6 SAYI:\n")
            f.write(str(best_6) + "\n")
        
        print(f"\n💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/superloto_12_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 SÜPER LOTO 6/12 BOTU")
    print("   60 sayı içinden en güçlü 12 sayı")
    print("   Kombinasyon: C(12,6) = 924")
    print("=" * 60)
    
    bot = SuperLoto12Predictor()
    bot.load_data()
    bot.run_backtest(test_size=50)
    result = bot.print_report()
    bot.save_results(result['final_12'], result['best_6'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
