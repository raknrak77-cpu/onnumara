"""
Tahmin motoru - Backtest ve ileri tahmin yapar
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from datetime import timedelta

class PredictionEngine:
    def __init__(self, excel_path="onnumara_2020.xlsx"):
        self.excel_path = excel_path
        self.df = None
        self.number_columns = [f'no-{i}' for i in range(1, 23)]
        
    def load_and_clean(self):
        """Excel'den veri yükle ve temizle - Sayfa7 için özel"""
        self.df = pd.read_excel(self.excel_path, sheet_name="Sayfa7", header=3)
        
        # Sütun isimlerini düzenle
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        # İlk gereksiz sütunu sil (genelde Unnamed)
        first_col = self.df.columns[0]
        if 'unnamed' in first_col or first_col == '':
            self.df = self.df.drop(columns=[first_col])
        
        # Tarih sütununu bul ve işle
        for col in self.df.columns:
            if 'çekiliş' in col or 'tarih' in col:
                # "1.Çekiliş[1] 03/08/2020" formatını ayıkla
                self.df['tarih_raw'] = self.df[col].astype(str)
                self.df['tarih'] = pd.to_datetime(
                    self.df['tarih_raw'].str.split().str[-1], 
                    format='%d/%m/%Y', 
                    errors='coerce'
                )
                self.df = self.df.drop(columns=[col, 'tarih_raw'])
                break
        
        # Çekiliş numarası ekle (index olarak)
        self.df['no'] = range(1, len(self.df) + 1)
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # NaN satırları temizle
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        
        print(f"✅ Yüklendi: {len(self.df)} çekiliş, {self.df['no'].min()}-{self.df['no'].max()}")
        return self.df
    
    def get_all_numbers_series(self):
        """Tüm sayıları tek liste olarak döndür"""
        if self.df is None:
            self.load_and_clean()
        all_nums = []
        for col in self.number_columns:
            all_nums.extend(self.df[col].tolist())
        return all_nums
    
    def estimate_next_date(self, offset=1):
        """Bir sonraki çekiliş tarihini tahmin et (Pazartesi/Cuma düzeni)"""
        if self.df is None:
            self.load_and_clean()
        last_date = self.df['tarih'].max()
        # Son çekilişten 3 veya 4 gün sonra
        next_date = last_date + timedelta(days=3 + (offset-1)*4)
        return next_date.strftime('%d.%m.%Y')
    
    def run_backtest(self, train_size=538, test_size=50):
        """Geriye dönük test"""
        if self.df is None:
            self.load_and_clean()
            
        results = []
        
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= len(self.df):
                break
                
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual_numbers = [test_row[col] for col in self.number_columns]
            
            predictions = {
                'frequency_based': self.frequency_prediction(train_df, 10),
                'markov_based': self.markov_prediction(train_df, 10),
                'monte_carlo': self.monte_carlo_prediction(train_df, 10),
                'random': self.random_prediction(10)
            }
            
            # Ensemble: 3 modelin ortak önerdiği sayılar
            all_pred_sets = [set(p) for p in [predictions['frequency_based'], 
                                               predictions['markov_based'], 
                                               predictions['monte_carlo']]]
            predictions['ensemble'] = list(set.intersection(*all_pred_sets)) if all_pred_sets else []
            
            score = len(set(predictions['ensemble']) & set(actual_numbers)) if predictions['ensemble'] else 0
            
            results.append({
                'cekilis_no': train_end + 1,
                'tarih': str(test_row['tarih'].date()),
                'gercekler': actual_numbers,
                'tahminler': predictions,
                'basarisayisi': score
            })
        
        return results
    
    def predict_future(self, n_predictions=3):
        """Gelecek çekilişler için tahmin"""
        if self.df is None:
            self.load_and_clean()
            
        predictions = []
        for i in range(n_predictions):
            pred = {
                'tahmin_no': i+1,
                'tarih_tahmini': self.estimate_next_date(i+1),
                'frequency_top10': self.frequency_prediction(self.df, 10),
                'markov_top10': self.markov_prediction(self.df, 10),
                'monte_carlo_top10': self.monte_carlo_prediction(self.df, 10),
                'random_top10': self.random_prediction(10)
            }
            # Ensemble
            all_sets = [set(pred['frequency_top10']), set(pred['markov_top10']), set(pred['monte_carlo_top10'])]
            pred['ensemble_top10'] = list(set.intersection(*all_sets)) if all_sets else []
            predictions.append(pred)
            
        return predictions
    
    # ========== MODEL 1: Frekans bazlı ==========
    def frequency_prediction(self, train_df, n=10):
        """En sık çıkan n sayıyı döndür"""
        all_nums = train_df[self.number_columns].values.flatten()
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ========== MODEL 2: Markov zinciri ==========
    def markov_prediction(self, train_df, n=10):
        """Bir sayıdan sonra hangi sayı gelme ihtimali yüksek?"""
        transitions = {}
        
        for _, row in train_df.iterrows():
            nums = row[self.number_columns].values
            for j in range(len(nums)-1):
                current = nums[j]
                next_num = nums[j+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        
        last_row = train_df.iloc[-1][self.number_columns].values
        last_num = last_row[-1]
        
        if last_num in transitions:
            next_nums = transitions[last_num]
            counter = Counter(next_nums)
            return [num for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
    # ========== MODEL 3: Monte Carlo ==========
    def monte_carlo_prediction(self, train_df, n=10, simulations=10000):
        """Geçmiş dağılıma göre simülasyon"""
        all_nums = train_df[self.number_columns].values.flatten()
        
        probs = {}
        for num in range(1, 81):
            probs[num] = (list(all_nums).count(num) + 1) / (len(all_nums) + 80)
        
        simulated_counts = Counter()
        for _ in range(simulations):
            selected = np.random.choice(
                list(probs.keys()), 
                size=22, 
                replace=False, 
                p=list(probs.values())
            )
            simulated_counts.update(selected)
        
        return [num for num, _ in simulated_counts.most_common(n)]
    
    # ========== MODEL 4: Rastgele ==========
    def random_prediction(self, n=10):
        """Tamamen rastgele n sayı seç (baz model)"""
        return sorted(np.random.choice(range(1, 81), n, replace=False).tolist())
    
    def save_results(self, results, filename):
        """Sonuçları JSON'a kaydet"""
        import os
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Kaydedildi: outputs/{filename}")
