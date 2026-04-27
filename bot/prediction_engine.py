import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import Counter

class PredictionEngine:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None
        
    def load_and_clean(self):
        self.df = pd.read_excel(self.excel_path)
        # ... data_loader.py'deki temizlik kodları
        
    def run_backtest(self, train_size=538, test_size=50):
        """Geriye dönük test"""
        results = []
        
        for i in range(test_size):
            train_end = train_size + i
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual_numbers = [test_row[f'no-{j}'] for j in range(1, 23)]
            
            # 5 farklı model ile tahmin
            predictions = {
                'frequency_based': self.frequency_prediction(train_df, 10),
                'markov_based': self.markov_prediction(train_df, 10),
                'ml_based': self.ml_prediction(train_df, 10),
                'monte_carlo': self.monte_carlo_prediction(train_df, 10),
                'ensemble': None  # Model karışımı
            }
            
            # Ensemble = 5 modelin ortak önerdiği sayılar
            all_preds = [set(p) for p in predictions.values() if p]
            predictions['ensemble'] = list(set.intersection(*all_preds)) if all_preds else []
            
            # Başarı skoru hesapla
            score = self.calculate_accuracy(predictions['ensemble'], actual_numbers)
            
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
        # Tüm veriyi kullan
        all_numbers = self.get_all_numbers_series()
        
        predictions = []
        for i in range(n_predictions):
            pred = {
                'tahmin_no': i+1,
                'tarih_tahmini': self.estimate_next_date(i+1),
                'frequency_top10': self.frequency_prediction(self.df, 10),
                'markov_top10': self.markov_prediction(self.df, 10),
                'monte_carlo_top10': self.monte_carlo_prediction(self.df, 10),
                'ml_top10': self.ml_prediction(self.df, 10),
                'ensemble_top10': None
            }
            # Ensemble hesapla...
            predictions.append(pred)
            
        return predictions
    
    # ========== MODEL 1: Frekans bazlı ==========
    def frequency_prediction(self, train_df, n=10):
        """En sık çıkan n sayıyı döndür"""
        number_cols = [f'no-{i}' for i in range(1, 23)]
        all_nums = train_df[number_cols].values.flatten()
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ========== MODEL 2: Markov zinciri ==========
    def markov_prediction(self, train_df, n=10):
        """Bir sayıdan sonra hangi sayı gelme ihtimali yüksek?"""
        number_cols = [f'no-{i}' for i in range(1, 23)]
        transitions = {}
        
        for _, row in train_df.iterrows():
            nums = row[number_cols].values
            for i in range(len(nums)-1):
                current = nums[i]
                next_num = nums[i+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        
        # Son çekilişin son sayısından başla
        last_row = train_df.iloc[-1][number_cols].values
        last_num = last_row[-1]
        
        if last_num in transitions:
            next_nums = transitions[last_num]
            counter = Counter(next_nums)
            return [num for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
    # ========== MODEL 3: Monte Carlo ==========
    def monte_carlo_prediction(self, train_df, n=10, simulations=10000):
        """Geçmiş dağılıma göre simülasyon"""
        number_cols = [f'no-{i}' for i in range(1, 23)]
        all_nums = train_df[number_cols].values.flatten()
        
        # Her sayının olasılığını hesapla
        probs = {}
        for num in range(1, 81):
            probs[num] = (list(all_nums).count(num) + 1) / (len(all_nums) + 80)
        
        # Simülasyon yap
        simulated_counts = Counter()
        for _ in range(simulations):
            # 22 sayı seç (tekrarsız, ağırlıklı)
            selected = np.random.choice(
                list(probs.keys()), 
                size=22, 
                replace=False, 
                p=list(probs.values())
            )
            simulated_counts.update(selected)
        
        return [num for num, _ in simulated_counts.most_common(n)]
    
    # ========== MODEL 4: Makine Öğrenmesi (XGBoost) ==========
    def ml_prediction(self, train_df, n=10):
        """Zaman serisi özellikleri ile XGBoost"""
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            
            # Özellik mühendisliği
            features = []
            labels = []
            
            # Son 5 çekilişin sayılarını özellik olarak kullan
            for idx in range(5, len(train_df)):
                window = train_df.iloc[idx-5:idx]
                window_numbers = []
                for _, row in window.iterrows():
                    window_numbers.extend([row[f'no-{j}'] for j in range(1, 23)])
                
                features.append(window_numbers)
                labels.extend([train_df.iloc[idx][f'no-{j}'] for j in range(1, 23)])
            
            # Model eğit (basitleştirilmiş)
            # ... XGBoost ile eğitim
            
            # Şimdilik frekans'a düş
            return self.frequency_prediction(train_df, n)
        except:
            return self.frequency_prediction(train_df, n)
    
    def calculate_accuracy(self, predicted, actual):
        """Kaç sayı doğru tahmin edildi?"""
        if not predicted:
            return 0
        return len(set(predicted) & set(actual))
    
    def save_results(self, results, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
