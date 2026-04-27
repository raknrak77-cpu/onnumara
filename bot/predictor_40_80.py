#!/usr/bin/env python3
"""
ON NUMARA 40-80 STRATEJİ BOTU
80 sayı yerine sadece 40-80 aralığına odaklanır
En sık çıkan 13 sayıyı önerir (çünkü 40-80'den ortalama 13 sayı çıkıyor)
Backtest ile en iyi stratejiyi bulur
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
        """40-80 arasındaki sayıları döndür"""
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    num = int(row[col])
                    if 40 <= num <= 80:  # SADECE 40-80 ARASI
                        nums.append(num)
                except:
                    pass
        return nums
    
    def get_all_numbers_in_range(self, row):
        """Tüm sayıları döndür (backtest için)"""
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums


# ============================================================
# MODELLER (Sadece 40-80 arası)
# ============================================================

class Models:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.range_start = 40
        self.range_end = 80
        
    def get_range_numbers(self):
        """40-80 arasındaki tüm sayılar"""
        return list(range(self.range_start, self.range_end + 1))
    
    # Model 1: Son N çekilişte en sık çıkanlar
    def recent(self, n=13, window=5):
        recent_nums = []
        window = min(window, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        
        # 40-80 arasındaki tüm sayıları başlat
        all_range_nums = self.get_range_numbers()
        counter = Counter(recent_nums)
        
        # Önce en sık çıkanları al
        result = [num for num, _ in counter.most_common(n)]
        
        # Eksik varsa, hiç çıkmayanlardan tamamla
        if len(result) < n:
            remaining = [num for num in all_range_nums if num not in result]
            result.extend(remaining[:n - len(result)])
        
        return result[:n]
    
    # Model 2: Tüm zamanların en sık çıkanları
    def frequency(self, n=13):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        all_range_nums = self.get_range_numbers()
        counter = Counter(all_nums)
        
        result = [num for num, _ in counter.most_common(n)]
        
        if len(result) < n:
            remaining = [num for num in all_range_nums if num not in result]
            result.extend(remaining[:n - len(result)])
        
        return result[:n]
    
    # Model 3: En uzun süredir çıkmayanlar (Due)
    def due(self, n=13):
        last_seen = {num: 0 for num in self.get_range_numbers()}
        
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in self.get_range_numbers()}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in due_sorted[:n]]
    
    # Model 4: Ağırlıklı frekans (yeni çekilişler daha ağırlıklı)
    def weighted(self, n=13, decay=0.95):
        weighted_counts = {num: 0 for num in self.get_range_numbers()}
        
        for idx, row in self.df.iterrows():
            weight = decay ** (len(self.df) - idx - 1)
            for num in self.get_numbers(row):
                weighted_counts[num] = weighted_counts.get(num, 0) + weight
        
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # Model 5: Trend analizi (yükselen sayılar)
    def trend(self, n=13, window=20):
        scores = {}
        
        for num in self.get_range_numbers():
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
# BACKTEST ve OPTİMİZASYON
# ============================================================

class Backtest:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_range_numbers_func = lambda row: get_numbers_func(row)
        self.models = Models(df, get_numbers_func)
        self.results = {}
        
    def test_model(self, model_func, model_name, test_size=50, **kwargs):
        """Tek bir modeli backtest et - 40-80 arası doğruluk"""
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
            
            # Test satırındaki 40-80 arası sayılar (doğru cevaplar)
            actual_40_80 = set(self.get_numbers(test_row))
            
            # Geçici model ile tahmin yap
            temp_models = Models(train_df, self.get_numbers)
            
            if model_name == 'recent':
                window = kwargs.get('window', 5)
                preds = temp_models.recent(13, window)
            elif model_name == 'frequency':
                preds = temp_models.frequency(13)
            elif model_name == 'due':
                preds = temp_models.due(13)
            elif model_name == 'weighted':
                decay = kwargs.get('decay', 0.95)
                preds = temp_models.weighted(13, decay)
            elif model_name == 'trend':
                window = kwargs.get('window', 20)
                preds = temp_models.trend(13, window)
            else:
                preds = []
            
            correct = len(set(preds) & actual_40_80)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def find_best_window_for_recent(self, test_size=50):
        """Son N çekiliş için en iyi pencere boyutunu bul"""
        print("  🔍 Son N çekiliş modeli optimize ediliyor (40-80)...")
        best_score = 0
        best_window = 5
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        
        for window in windows:
            if window > len(self.df) - test_size:
                continue
            score = self.test_model(None, 'recent', test_size, window=window)
            print(f"     window={window:2d} -> {score:.2f}/13 doğru")
            if score > best_score:
                best_score = score
                best_window = window
        
        print(f"  ✅ En iyi pencere: {best_window} (ortalama {best_score:.2f}/13)")
        return best_window, best_score
    
    def find_best_decay_for_weighted(self, test_size=50):
        """Ağırlıklı frekans için en iyi decay değerini bul"""
        print("  🔍 Ağırlıklı frekans modeli optimize ediliyor (40-80)...")
        best_score = 0
        best_decay = 0.95
        decays = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
        
        for decay in decays:
            score = self.test_model(None, 'weighted', test_size, decay=decay)
            print(f"     decay={decay} -> {score:.2f}/13 doğru")
            if score > best_score:
                best_score = score
                best_decay = decay
        
        print(f"  ✅ En iyi decay: {best_decay} (ortalama {best_score:.2f}/13)")
        return best_decay, best_score
    
    def run_all_backtests(self, test_size=50):
        """Tüm modelleri backtest et ve karşılaştır"""
        print(f"\n📊 BACKTEST BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("   Hedef: 40-80 arası sayılar")
        print("-" * 50)
        
        # Basit modeller
        print("\n  📈 TEMEL MODELLER:")
        models_to_test = ['frequency', 'due']
        for model in models_to_test:
            score = self.test_model(None, model, test_size)
            self.results[model] = score
            print(f"     {model}: {score:.2f}/13 doğru")
        
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
        print("🏆 MODEL BAŞARI SIRALAMASI (40-80 arası 13'te kaç doğru):")
        print("-" * 50)
        
        sorted_results = sorted([(k, v) for k, v in self.results.items() if isinstance(v, (int, float))], 
                                key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_results):
            stars = "⭐" * int(score // 1) + "☆" * (13 - int(score // 1))
            print(f"  {i+1}. {model:12}: {score:.2f}/13 {stars}")
        
        return self.results


# ============================================================
# ANA TAHMİN SINIFI
# ============================================================

class Predictor40_80:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.backtest_results = None
        self.optimal_settings = {}
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers_40_80 = loader.get_numbers
        self.get_all_numbers = loader.get_all_numbers_in_range
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        
        # 40-80 arası istatistik
        total_40_80_count = 0
        for _, row in self.df.iterrows():
            total_40_80_count += len(self.get_numbers_40_80(row))
        avg_per_draw = total_40_80_count / len(self.df)
        print(f"📊 40-80 arası ortalama çıkan sayı: {avg_per_draw:.1f}/22")
        
        return self.df
    
    def run_backtest(self, test_size=50):
        bt = Backtest(self.df, self.get_numbers_40_80)
        self.backtest_results = bt.run_all_backtests(test_size)
        
        self.optimal_settings = {
            'best_model': max([(k, v) for k, v in self.backtest_results.items() if isinstance(v, (int, float))], key=lambda x: x[1])[0],
            'recent_window': self.backtest_results.get('recent_best_window', 5),
            'weighted_decay': self.backtest_results.get('weighted_best_decay', 0.95)
        }
        
        print(f"\n🎯 EN İYİ MODEL: {self.optimal_settings['best_model']}")
        return self.backtest_results
    
    def get_optimized_predictions(self, n=13):
        """Optimize edilmiş ayarlarla tahmin yap"""
        
        recent_window = self.optimal_settings.get('recent_window', 5)
        weighted_decay = self.optimal_settings.get('weighted_decay', 0.95)
        
        temp_models = Models(self.df, self.get_numbers_40_80)
        
        # Her modelden tahmin al
        predictions = {
            'recent': temp_models.recent(n*2, recent_window),
            'frequency': temp_models.frequency(n*2),
            'due': temp_models.due(n*2),
            'weighted': temp_models.weighted(n*2, weighted_decay),
            'trend': temp_models.trend(n*2)
        }
        
        # Backtest başarısına göre ağırlıklar
        weights = {
            'recent': self.backtest_results.get('recent', 1.0) if self.backtest_results else 1.0,
            'frequency': self.backtest_results.get('frequency', 1.0) if self.backtest_results else 1.0,
            'due': self.backtest_results.get('due', 1.0) if self.backtest_results else 1.0,
            'weighted': self.backtest_results.get('weighted', 1.0) if self.backtest_results else 1.0,
            'trend': self.backtest_results.get('trend', 1.0) if self.backtest_results else 1.0
        }
        
        # Ağırlıklı skor
        weighted_scores = {}
        for model_name, preds in predictions.items():
            weight = weights.get(model_name, 1.0)
            for rank, num in enumerate(preds):
                score = (len(preds) - rank) * weight
                weighted_scores[num] = weighted_scores.get(num, 0) + score
        
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        final_13 = [num for num, _ in sorted_scores[:n]]
        
        return {
            'final_13': final_13,
            'models': predictions,
            'weights': weights,
            'best_model': self.optimal_settings.get('best_model', 'unknown')
        }
    
    def print_report(self):
        if self.backtest_results is None:
            self.run_backtest()
        
        result = self.get_optimized_predictions()
        
        print("\n" + "=" * 60)
        print("🎯 ON NUMARA 40-80 STRATEJİ RAPORU")
        print("   Sadece 40-80 arası sayılara odaklanır")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🎯 En İyi Model: {result['best_model']}")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 13 SAYI (40-80 arası)")
        print("-" * 60)
        
        final = result['final_13']
        for i in range(0, len(final), 5):
            group = final[i:i+5]
            print(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("📊 MODELLERİN İLK 10 TAHMİNİ (40-80 arası)")
        print("-" * 60)
        
        for model_name, preds in result['models'].items():
            print(f"\n  {model_name.upper()}:")
            print(f"    {preds[:10]}")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu 13 sayı, 40-80 aralığında en güçlü adaylardır.")
        print("   Her çekilişte bu aralıktan ortalama 13 sayı çıkmaktadır.")
        print("=" * 60)
        
        return result
    
    def save_results(self, result, filename="predictions_40_80.json"):
        os.makedirs('outputs', exist_ok=True)
        
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump({
                'final_13': result['final_13'],
                'best_model': result['best_model'],
                'weights': result['weights'],
                'backtest_results': self.backtest_results
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/report_40_80.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 ON NUMARA 40-80 STRATEJİ RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"En İyi Model: {result['best_model']}\n\n")
            f.write("EN GÜÇLÜ 13 SAYI:\n")
            f.write(str(result['final_13']) + "\n\n")
            f.write("BACKTEST SONUÇLARI:\n")
            for model, score in self.backtest_results.items():
                if isinstance(score, (int, float)):
                    f.write(f"  {model}: {score:.2f}/13\n")
        
        print(f"\n💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/report_40_80.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ON NUMARA 40-80 STRATEJİ BOTU")
    print("   80 sayı yerine sadece 40-80 aralığı (41 sayı)")
    print("   Backtest ile en iyi 13 sayıyı bulur")
    print("=" * 60)
    
    bot = Predictor40_80()
    bot.load_data()
    bot.run_backtest(test_size=50)
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
