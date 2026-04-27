"""
Tahmin Motoru - Excel yapınıza tam uyumlu
Excel sütunları: no, tarih, no_1, no_2, ..., no_22
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import json
from datetime import timedelta
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)


class PredictionEngine:
    def __init__(self, excel_path: str = "onnumara_2020.xlsx", sheet_name: str = "s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        
    def load_data(self):
        """Veriyi yükle - Başlık satırı ilk satırda"""
        
        # header=0 ile oku (ilk satır başlık olarak kullanılır)
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        print(f"📋 Sütunlar: {list(self.df.columns)}")
        
        # Tarih sütununu dönüştür (gg.aa.yyyy formatı)
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        else:
            # Alternatif: 'Tarih' veya 'DATE' olabilir
            for col in self.df.columns:
                if 'tarih' in str(col).lower():
                    self.df['tarih'] = pd.to_datetime(self.df[col], format='%d.%m.%Y', errors='coerce')
                    break
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # NaN satırları temizle
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 İlk tarih: {self.df['tarih'].min().strftime('%d.%m.%Y')}")
            print(f"📅 Son tarih: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        
        return self.df
    
    def get_valid_numbers(self, row) -> List[int]:
        """Bir satırdan geçerli sayıları al"""
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums
    
    # ==================== MODEL 1: FREKANS BAZLI ====================
    def frequency_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend([int(x) for x in train_df[col].dropna().tolist()])
        
        if not all_nums:
            return list(range(1, n+1))
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== MODEL 2: GECİKME BAZLI ====================
    def due_numbers_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        last_seen = {num: 0 for num in range(1, 81)}
        
        for idx, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for num in nums:
                last_seen[num] = idx
        
        last_idx = len(train_df) - 1 if len(train_df) > 0 else 0
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # ==================== MODEL 3: MARKOV ZİNCİRİ ====================
    def markov_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        if len(train_df) == 0:
            return self.frequency_prediction(train_df, n)
        
        transitions = {}
        for _, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for i in range(len(nums)-1):
                current, next_num = nums[i], nums[i+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        
        last_row = train_df.iloc[-1]
        last_nums = self.get_valid_numbers(last_row)
        
        if not last_nums:
            return self.frequency_prediction(train_df, n)
        
        last_num = last_nums[-1]
        
        if last_num in transitions:
            counter = Counter(transitions[last_num])
            return [num for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
    # ==================== MODEL 4: MONTE CARLO ====================
    def monte_carlo_prediction(self, train_df: pd.DataFrame, n: int = 10, simulations: int = 5000) -> List[int]:
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend([int(x) for x in train_df[col].dropna().tolist()])
        
        if not all_nums:
            return list(range(1, n+1))
        
        probs = {}
        total = len(all_nums)
        for num in range(1, 81):
            count = all_nums.count(num)
            probs[num] = (count + 1) / (total + 80)
        
        simulated_counts = Counter()
        for _ in range(simulations):
            selected = np.random.choice(
                list(probs.keys()), size=22, replace=False, p=list(probs.values())
            )
            simulated_counts.update(selected)
        
        return [num for num, _ in simulated_counts.most_common(n)]
    
    # ==================== MODEL 5: YAKIN GEÇMİŞ ====================
    def recent_prediction(self, train_df: pd.DataFrame, n: int = 10, window: int = 5) -> List[int]:
        recent_nums = []
        start_idx = max(0, len(train_df) - window)
        for idx in range(start_idx, len(train_df)):
            row_nums = self.get_valid_numbers(train_df.iloc[idx])
            recent_nums.extend(row_nums)
        
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== ENSEMBLE ====================
    def ensemble_prediction(self, train_df: pd.DataFrame, models: List[str] = None, n: int = 10) -> List[int]:
        if models is None:
            models = ['frequency', 'markov', 'monte_carlo', 'due']
        
        model_funcs = {
            'frequency': self.frequency_prediction,
            'markov': self.markov_prediction,
            'monte_carlo': self.monte_carlo_prediction,
            'due': self.due_numbers_prediction,
            'recent': self.recent_prediction
        }
        
        all_predictions = []
        for model in models:
            if model in model_funcs:
                pred = model_funcs[model](train_df, n*2)
                all_predictions.extend(pred)
        
        counter = Counter(all_predictions)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== BACKTEST ====================
    def run_backtest(self, train_size: int = 500, test_size: int = 50) -> Dict:
        if self.df is None:
            self.load_data()
        
        total = len(self.df)
        if total == 0:
            print("❌ Veri bulunamadı!")
            return {}
        
        train_size = min(train_size, total - test_size)
        
        if train_size <= 0:
            print(f"⚠️ Yetersiz veri! Total: {total}, Train: {train_size}, Test: {test_size}")
            return {}
        
        models = ['frequency', 'markov', 'monte_carlo', 'due', 'recent', 'ensemble']
        results = {model: {'scores': [], 'total_correct': 0, 'total_tested': 0} for model in models}
        
        print(f"📊 Backtest: {train_size} eğitim, {test_size} test")
        
        for i in range(min(test_size, total - train_size)):
            train_end = train_size + i
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual = set(self.get_valid_numbers(test_row))
            
            predictions = {
                'frequency': self.frequency_prediction(train_df, 10),
                'markov': self.markov_prediction(train_df, 10),
                'monte_carlo': self.monte_carlo_prediction(train_df, 10),
                'due': self.due_numbers_prediction(train_df, 10),
                'recent': self.recent_prediction(train_df, 10),
                'ensemble': self.ensemble_prediction(train_df, n=10)
            }
            
            for model, preds in predictions.items():
                correct = len(set(preds) & actual)
                results[model]['scores'].append(correct)
                results[model]['total_correct'] += correct
                results[model]['total_tested'] += 1
        
        for model in results:
            if results[model]['total_tested'] > 0:
                results[model]['avg_score'] = results[model]['total_correct'] / results[model]['total_tested']
            else:
                results[model]['avg_score'] = 0
        
        print("\n📈 Backtest Sonuçları:")
        for model in sorted(results.keys(), key=lambda x: results[x]['avg_score'], reverse=True):
            print(f"   {model:12}: {results[model]['avg_score']:.2f}/10 doğru")
        
        return results
    
    # ==================== İLERİ TAHMİN ====================
    def predict_future(self, n_predictions: int = 3) -> List[Dict]:
        if self.df is None:
            self.load_data()
        
        if len(self.df) == 0:
            return []
        
        future_predictions = []
        last_date = self.df['tarih'].max()
        
        freq_pred = self.frequency_prediction(self.df, 10)
        markov_pred = self.markov_prediction(self.df, 10)
        monte_pred = self.monte_carlo_prediction(self.df, 10)
        due_pred = self.due_numbers_prediction(self.df, 10)
        ensemble_pred = self.ensemble_prediction(self.df, n=10)
        
        for i in range(n_predictions):
            next_date = last_date + timedelta(days=3 + i*4)
            
            pred = {
                'tahmin_no': i + 1,
                'tarih': next_date.strftime('%d.%m.%Y'),
                'frequency_top10': freq_pred,
                'markov_top10': markov_pred,
                'monte_carlo_top10': monte_pred,
                'due_numbers_top10': due_pred,
                'ensemble_top10': ensemble_pred
            }
            future_predictions.append(pred)
        
        return future_predictions
    
    def save_results(self, results, filename: str):
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"💾 Kaydedildi: outputs/{filename}")
