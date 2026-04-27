"""
Tahmin Motoru - Temel modeller
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import json
from datetime import timedelta
import os
import re

class NumpyEncoder(json.JSONEncoder):
    """NumPy tiplerini JSON'a dönüştür"""
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
        """Veriyi yükle ve temizle"""
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        
        print(f"📋 Sütunlar: {list(self.df.columns)}")
        
        # Tarih sütununu bul (farklı isimlendirmeler için)
        date_column = None
        for col in self.df.columns:
            col_lower = str(col).lower()
            if 'tarih' in col_lower or 'date' in col_lower:
                date_column = col
                break
        
        # Eğer tarih sütunu bulunamazsa, ilgili sütunu manuel dene
        if date_column is None:
            # 'tarih' veya 'Tarih' olabilir
            if 'tarih' in self.df.columns:
                date_column = 'tarih'
            elif 'Tarih' in self.df.columns:
                date_column = 'Tarih'
            elif 'DATE' in self.df.columns:
                date_column = 'DATE'
            else:
                # İkinci sütun genelde tarih oluyor
                date_column = self.df.columns[1]
        
        print(f"📅 Tarih sütunu: {date_column}")
        
        # Tarihi dönüştür - Türkiye formatı gg.aa.yyyy
        try:
            self.df['tarih'] = pd.to_datetime(self.df[date_column], format='%d.%m.%Y', errors='coerce')
        except:
            try:
                self.df['tarih'] = pd.to_datetime(self.df[date_column], format='%d/%m/%Y', errors='coerce')
            except:
                self.df['tarih'] = pd.to_datetime(self.df[date_column], errors='coerce')
        
        # Sayı sütunlarını bul
        number_cols = []
        for col in self.df.columns:
            col_str = str(col).lower()
            # no_1, no1, no-1, 1, vs. formatları
            if re.match(r'no[_\-\s]?\d+', col_str) or re.match(r'^\d+$', col_str):
                number_cols.append(col)
            # Ayrıca düz sayı sütunları (1,2,3...)
            try:
                if 1 <= int(col_str) <= 22:
                    number_cols.append(col)
            except:
                pass
        
        # Eğer otomatik bulamadıysa, no_1'den no_22'ye kadar dene
        if len(number_cols) < 22:
            number_cols = []
            for i in range(1, 23):
                for candidate in [f'no_{i}', f'no{i}', f'no-{i}', f'{i}']:
                    if candidate in self.df.columns:
                        number_cols.append(candidate)
                        break
        
        self.number_columns = number_cols[:22]
        print(f"🔢 Sayı sütunları ({len(self.number_columns)} adet): {self.number_columns[:5]}...")
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # NaN satırları temizle
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 İlk tarih: {self.df['tarih'].min()}")
        print(f"📅 Son tarih: {self.df['tarih'].max()}")
        
        return self.df
    
    # ==================== MODEL 1: FREKANS BAZLI ====================
    def frequency_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """En sık çıkan n sayı"""
        all_nums = train_df[self.number_columns].values.flatten()
        all_nums = [x for x in all_nums if pd.notna(x)]
        counter = Counter(all_nums)
        return [int(num) for num, _ in counter.most_common(n)]
    
    # ==================== MODEL 2: GECİKME BAZLI (Due Numbers) ====================
    def due_numbers_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """En uzun süredir çıkmayan sayılar"""
        all_nums = train_df[self.number_columns].values.flatten()
        all_nums = [x for x in all_nums if pd.notna(x)]
        
        last_seen = {num: 0 for num in range(1, 81)}
        
        for idx, row in train_df.iterrows():
            for num in row[self.number_columns]:
                if pd.notna(num):
                    last_seen[int(num)] = idx
        
        last_idx = len(train_df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # ==================== MODEL 3: MARKOV ZİNCİRİ ====================
    def markov_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """Markov zinciri - sayı geçiş olasılıkları"""
        transitions = {}
        
        for _, row in train_df.iterrows():
            nums = [int(x) for x in row[self.number_columns].values if pd.notna(x)]
            for i in range(len(nums)-1):
                current, next_num = nums[i], nums[i+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        
        last_row = train_df.iloc[-1][self.number_columns].values
        last_nums = [int(x) for x in last_row if pd.notna(x)]
        
        if not last_nums:
            return self.frequency_prediction(train_df, n)
        
        last_num = last_nums[-1]
        
        if last_num in transitions:
            counter = Counter(transitions[last_num])
            return [int(num) for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
    # ==================== MODEL 4: MONTE CARLO ====================
    def monte_carlo_prediction(self, train_df: pd.DataFrame, n: int = 10, simulations: int = 5000) -> List[int]:
        """Monte Carlo simülasyonu"""
        all_nums = []
        for col in self.number_columns:
            col_nums = train_df[col].dropna().tolist()
            all_nums.extend(col_nums)
        
        all_nums = [int(x) for x in all_nums if pd.notna(x)]
        
        if not all_nums:
            return list(range(1, n+1))
        
        # Olasılıkları hesapla (Laplace smoothing)
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
        
        return [int(num) for num, _ in simulated_counts.most_common(n)]
    
    # ==================== MODEL 5: ISI HARİTASI (Birliktelik) ====================
    def cooccurrence_prediction(self, train_df: pd.DataFrame, n: int = 10, window: int = 5) -> List[int]:
        """Birlikte çıkma analizi - son window çekilişte en sık görülenler"""
        recent_nums = []
        start_idx = max(0, len(train_df) - window)
        for idx in range(start_idx, len(train_df)):
            row_nums = train_df.iloc[idx][self.number_columns].values
            recent_nums.extend([int(x) for x in row_nums if pd.notna(x)])
        
        counter = Counter(recent_nums)
        return [int(num) for num, _ in counter.most_common(n)]
    
    # ==================== MODEL 6: ARALIK ANALİZİ ====================
    def interval_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """Sayılar arasındaki farkları analiz et"""
        intervals = []
        for _, row in train_df.iterrows():
            nums = sorted([int(x) for x in row[self.number_columns].values if pd.notna(x)])
            for i in range(len(nums)-1):
                intervals.append(nums[i+1] - nums[i])
        
        avg_interval = sum(intervals) / len(intervals) if intervals else 4
        
        last_row = sorted([int(x) for x in train_df.iloc[-1][self.number_columns].values if pd.notna(x)])
        if not last_row:
            return self.frequency_prediction(train_df, n)
        
        predictions = []
        last_num = last_row[-1]
        
        for _ in range(n):
            next_num = last_num + int(avg_interval)
            if next_num > 80:
                next_num = next_num - 80
            if next_num < 1:
                next_num = 1
            predictions.append(next_num)
            last_num = next_num
        
        return predictions
    
    # ==================== ENSEMBLE ====================
    def ensemble_prediction(self, train_df: pd.DataFrame, models: List[str] = None, n: int = 10) -> List[int]:
        """Birden fazla modelin ortak önerileri"""
        if models is None:
            models = ['frequency', 'markov', 'monte_carlo', 'due']
        
        model_funcs = {
            'frequency': self.frequency_prediction,
            'markov': self.markov_prediction,
            'monte_carlo': self.monte_carlo_prediction,
            'due': self.due_numbers_prediction,
            'cooccurrence': self.cooccurrence_prediction
        }
        
        all_predictions = []
        for model in models:
            if model in model_funcs:
                pred = model_funcs[model](train_df, n*2)
                all_predictions.extend(pred)
        
        counter = Counter(all_predictions)
        return [int(num) for num, _ in counter.most_common(n)]
    
    # ==================== BACKTEST ====================
    def run_backtest(self, train_size: int = 500, test_size: int = 50) -> Dict:
        """Geriye dönük test - tüm modeller için"""
        if self.df is None:
            self.load_data()
        
        total = len(self.df)
        train_size = min(train_size, total - test_size)
        
        if train_size <= 0 or test_size <= 0:
            print(f"⚠️ Yetersiz veri! Total: {total}, Train: {train_size}, Test: {test_size}")
            return {}
        
        models = ['frequency', 'markov', 'monte_carlo', 'due', 'cooccurrence', 'ensemble']
        results = {model: {'scores': [], 'total_correct': 0, 'total_tested': 0} for model in models}
        
        print(f"📊 Backtest: {train_size} eğitim, {test_size} test")
        
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
                
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual = set([int(x) for x in test_row[self.number_columns].values if pd.notna(x)])
            
            predictions = {
                'frequency': self.frequency_prediction(train_df, 10),
                'markov': self.markov_prediction(train_df, 10),
                'monte_carlo': self.monte_carlo_prediction(train_df, 10),
                'due': self.due_numbers_prediction(train_df, 10),
                'cooccurrence': self.cooccurrence_prediction(train_df, 10),
                'ensemble': self.ensemble_prediction(train_df, n=10)
            }
            
            for model, preds in predictions.items():
                correct = len(set(preds) & actual)
                results[model]['scores'].append(correct)
                results[model]['total_correct'] += correct
                results[model]['total_tested'] += 1
        
        # Ortalamaları hesapla
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
        """Gelecek çekilişler için tahmin"""
        if self.df is None:
            self.load_data()
        
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
        """Sonuçları kaydet"""
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"💾 Kaydedildi: outputs/{filename}")
