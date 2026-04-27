#!/usr/bin/env python3
"""
ON NUMARA 22'LİK SET TAHMİN BOTU
Tek dosya - Bağımsız çalışır
80 sayı içinden en güçlü 22 sayıyı önerir
GERİYE DÖNÜK TEST ile EN İYİ STRATEJİYİ BULUR
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
    
    # Model 1: Son N çekilişte en sık çıkanlar
    def recent(self, n=22, window=5):
        recent_nums = []
        window = min(window, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        result = [num for num, _ in counter.most_common(n)]
        # Eksik varsa frekanstan tamamla
        if len(result) < n:
            freq = self.frequency(n*2)
            for num in freq:
                if num not in result:
                    result.append(num)
                if len(result) == n:
                    break
        return result[:n]
    
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
# BACKTEST ve OPTİMİZASYON
# ============================================================

class Backtest:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.models = Models(df, get_numbers_func)
        self.results = {}
        
    def test_model(self, model_func, model_name, test_size=50, **kwargs):
        """Tek bir modeli backtest et"""
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
            
            # Geçici model ile tahmin yap
            temp_models = Models(train_df, self.get_numbers)
            
            if model_name == 'recent':
                window = kwargs.get('window', 5)
                preds = temp_models.recent(22, window)
            elif model_name == 'frequency':
                preds = temp_models.frequency(22)
            elif model_name == 'due':
                preds = temp_models.due(22)
            elif model_name == 'weighted':
                decay = kwargs.get('decay', 0.95)
                preds = temp_models.weighted(22, decay)
            elif model_name == 'trend':
                window = kwargs.get('window', 20)
                preds = temp_models.trend(22, window)
            elif model_name == 'physical':
                preds = temp_models.physical(22)
            else:
                preds = []
            
            correct = len(set(preds) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def find_best_window_for_recent(self, test_size=50):
        """Son N çekiliş için en iyi pencere boyutunu bul"""
        print("  🔍 Son N çekiliş modeli optimize ediliyor...")
        best_score = 0
        best_window = 5
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        
        for window in windows:
            if window > len(self.df) - test_size:
                continue
            score = self.test_model(None, 'recent', test_size, window=window)
            print(f"     window={window:2d} -> {score:.2f}/22 doğru")
            if score > best_score:
                best_score = score
                best_window = window
        
        print(f"  ✅ En iyi pencere: {best_window} (ortalama {best_score:.2f}/22)")
        return best_window, best_score
    
    def find_best_decay_for_weighted(self, test_size=50):
        """Ağırlıklı frekans için en iyi decay değerini bul"""
        print("  🔍 Ağırlıklı frekans modeli optimize ediliyor...")
        best_score = 0
        best_decay = 0.95
        decays = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
        
        for decay in decays:
            score = self.test_model(None, 'weighted', test_size, decay=decay)
            print(f"     decay={decay} -> {score:.2f}/22 doğru")
            if score > best_score:
                best_score = score
                best_decay = decay
        
        print(f"  ✅ En iyi decay: {best_decay} (ortalama {best_score:.2f}/22)")
        return best_decay, best_score
    
    def run_all_backtests(self, test_size=50):
        """Tüm modelleri backtest et ve karşılaştır"""
        print(f"\n📊 BACKTEST BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("-" * 50)
        
        # Basit modeller
        print("\n  📈 TEMEL MODELLER:")
        models_to_test = ['frequency', 'due', 'physical']
        for model in models_to_test:
            score = self.test_model(None, model, test_size)
            self.results[model] = score
            print(f"     {model}: {score:.2f}/22 doğru")
        
        # Optimizasyon gerektiren modeller
        best_window, recent_score = self.find_best_window_for_recent(test_size)
        self.results['recent'] = recent_score
        self.results['recent_best_window'] = best_window
        
        best_decay, weighted_score = self.find_best_decay_for_weighted(test_size)
        self.results['weighted'] = weighted_score
        self.results['weighted_best_decay'] = best_decay
        
        trend_score = self.test_model(None, 'trend', test_size)
        self.results['trend'] = trend_score
        
        # Sıralama
        print("\n" + "-" * 50)
        print("🏆 MODEL BAŞARI SIRALAMASI (22'de kaç doğru):")
        print("-" * 50)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
        for i, (model, score) in enumerate(sorted_results):
            if isinstance(score, (int, float)):
                stars = "⭐" * int(score // 2) + "☆" * (11 - int(score // 2))
                print(f"  {i+1}. {model:12}: {score:.2f}/22 {stars}")
        
        return self.results


# ============================================================
# ANA TAHMİN SINIFI (Backtest ile optimize edilmiş)
# ============================================================

class Predictor22:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.models = None
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
        """Backtest çalıştır ve en iyi stratejiyi bul"""
        bt = Backtest(self.df, self.get_numbers)
        self.backtest_results = bt.run_all_backtests(test_size)
        
        # Optimal ayarları kaydet
        self.optimal_settings = {
            'best_model': max([(k, v) for k, v in self.backtest_results.items() if isinstance(v, (int, float))], key=lambda x: x[1])[0],
            'recent_window': self.backtest_results.get('recent_best_window', 5),
            'weighted_decay': self.backtest_results.get('weighted_best_decay', 0.95)
        }
        
        print(f"\n🎯 EN İYİ MODEL: {self.optimal_settings['best_model']}")
        print(f"   Son {self.optimal_settings['recent_window']} çekiliş kullanılacak")
        
        return self.backtest_results
    
    def get_optimized_ensemble(self):
        """Optimize edilmiş ayarlarla ensemble oluştur"""
        
        # Optimal ayarları kullan
        recent_window = self.optimal_settings.get('recent_window', 5)
        weighted_decay = self.optimal_settings.get('weighted_decay', 0.95)
        
        # Geçici modeller sınıfı
        temp_models = Models(self.df, self.get_numbers)
        
        models_pred = {
            'recent': temp_models.recent(30, recent_window),
            'frequency': temp_models.frequency(30),
            'due': temp_models.due(30),
            'weighted': temp_models.weighted(30, weighted_decay),
            'trend': temp_models.trend(30),
            'physical': temp_models.physical(30)
        }
        
        # Backtest sonuçlarına göre ağırlıklar
        weights = {
            'recent': self.backtest_results.get('recent', 1.0),
            'frequency': self.backtest_results.get('frequency', 1.0),
            'due': self.backtest_results.get('due', 1.0),
            'weighted': self.backtest_results.get('weighted', 1.0),
            'trend': self.backtest_results.get('trend', 1.0),
            'physical': self.backtest_results.get('physical', 1.0)
        }
        
        # Ağırlıkları normalize et
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight * 10 for k, v in weights.items()}
        
        # Kesişim
        all_sets = [set(models_pred[m]) for m in models_pred]
        intersection = list(set.intersection(*all_sets))
        
        # Ağırlıklı skor
        weighted_scores = {}
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
            'scores': dict(sorted_scores[:30]),
            'weights': weights,
            'best_model': self.optimal_settings['best_model']
        }
    
    def print_report(self):
        # Önce backtest yap
        if self.backtest_results is None:
            self.run_backtest()
        
        result = self.get_optimized_ensemble()
        
        print("\n" + "=" * 60)
        print("🎯 ON NUMARA 22'LİK SET TAHMİN RAPORU")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🎯 En İyi Model: {result['best_model']}")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 22 SAYI (Optimize Ensemble)")
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
            print("  (Ortak sayı bulunamadı - modeller çeşitli)")
        
        print("\n" + "-" * 60)
        print("📊 MODELLERİN AĞIRLIKLARI (Backtest başarısına göre)")
        print("-" * 60)
        
        for model, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:12}: {weight:.2f}")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return result
    
    def save_results(self, result, filename="predictions_22.json"):
        os.makedirs('outputs', exist_ok=True)
        
        # JSON kaydet
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump({
                'final_22': result['final_22'],
                'best_model': result['best_model'],
                'weights': result['weights'],
                'backtest_results': self.backtest_results,
                'optimal_settings': self.optimal_settings
            }, f, ensure_ascii=False, indent=2)
        
        # TXT rapor kaydet
        with open('outputs/report_22.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 ON NUMARA 22'LİK SET TAHMİN RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"En İyi Model: {result['best_model']}\n\n")
            f.write("EN GÜÇLÜ 22 SAYI:\n")
            f.write(str(result['final_22']) + "\n\n")
            f.write("BACKTEST SONUÇLARI:\n")
            for model, score in self.backtest_results.items():
                if isinstance(score, (int, float)):
                    f.write(f"  {model}: {score:.2f}/22\n")
        
        print(f"💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/report_22.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ON NUMARA 22'LİK SET TAHMİN BOTU")
    print("   80 sayı içinden en güçlü 22 sayı")
    print("   Backtest ile EN İYİ STRATEJİYİ BULUR")
    print("=" * 60)
    
    bot = Predictor22()
    bot.load_data()
    bot.run_backtest(test_size=50)
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
