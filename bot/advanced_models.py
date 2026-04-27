"""
Gelişmiş Makine Öğrenmesi Modelleri
XGBoost, LSTM, Prophet, Apriori
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdvancedModels:
    def __init__(self, df: pd.DataFrame, number_columns: List[str]):
        self.df = df
        self.number_columns = number_columns
    
    # ==================== APRIORI (Birliktelik Kuralı) ====================
    def apriori_analysis(self, min_support: float = 0.05) -> List[Tuple]:
        """Sık geçen sayı kümelerini bul"""
        try:
            from mlxtend.frequent_patterns import apriori
            from mlxtend.preprocessing import TransactionEncoder
            
            transactions = []
            for _, row in self.df.iterrows():
                transactions.append(list(row[self.number_columns].values))
            
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            frequent = apriori(df_encoded, min_support=min_support, use_colnames=True)
            frequent['length'] = frequent['itemsets'].apply(len)
            frequent = frequent.sort_values('support', ascending=False)
            
            return frequent[['itemsets', 'support', 'length']].head(20).to_dict('records')
        except Exception as e:
            print(f"Apriori hatası: {e}")
            return []
    
    # ==================== RANDOM FOREST ====================
    def random_forest_prediction(self, n: int = 10) -> List[int]:
        """Random Forest ile tahmin"""
        features = []
        targets = []
        
        # Özellik mühendisliği: son 3 çekilişin sayıları
        for idx in range(3, len(self.df)):
            window = []
            for j in range(3):
                window.extend(self.df.iloc[idx-3+j][self.number_columns].values)
            features.append(window)
            targets.extend(self.df.iloc[idx][self.number_columns].values)
        
        if len(features) < 100:
            return list(range(1, n+1))
        
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Son çekilişi kullanarak tahmin
        last_window = []
        for j in range(3):
            last_window.extend(self.df.iloc[-(3-j)][self.number_columns].values)
        
        proba = rf.predict_proba([last_window])[0]
        top_n = np.argsort(proba)[-n:][::-1]
        
        return [int(rf.classes_[i]) for i in top_n]
    
    # ==================== ZAMAN SERİSİ ANALİZİ ====================
    def time_series_analysis(self) -> Dict:
        """Her sayının zaman içindeki trendi"""
        trends = {}
        for num in range(1, 81):
            appearances = []
            for idx, row in self.df.iterrows():
                if num in row[self.number_columns].values:
                    appearances.append(idx)
            
            if len(appearances) > 1:
                gaps = np.diff(appearances)
                trends[num] = {
                    'total': len(appearances),
                    'avg_gap': np.mean(gaps) if len(gaps) > 0 else len(self.df),
                    'last_seen': appearances[-1],
                    'trend': 'increasing' if len(gaps) > 2 and gaps[-1] < gaps[-2] else 'decreasing'
                }
            else:
                trends[num] = {'total': 0, 'avg_gap': len(self.df), 'last_seen': -1, 'trend': 'unknown'}
        
        return trends
    
    # ==================== AĞIRLIKLI SİMÜLASYON ====================
    def weighted_simulation(self, n: int = 10, simulations: int = 5000) -> List[int]:
        """Zamansal ağırlıklı Monte Carlo"""
        all_nums = []
        weights = []
        
        for idx, row in self.df.iterrows():
            weight = np.exp(-(len(self.df) - idx) / (len(self.df) / 3))  # Daha yeni çekilişler daha ağırlıklı
            for num in row[self.number_columns].values:
                all_nums.append(num)
                weights.append(weight)
        
        # Her sayının ağırlıklı olasılığı
        probs = {}
        total_weight = sum(weights)
        
        for num in range(1, 81):
            indices = [i for i, x in enumerate(all_nums) if x == num]
            probs[num] = (sum(weights[i] for i in indices) + 1) / (total_weight + 80)
        
        simulated_counts = Counter()
        for _ in range(simulations):
            selected = np.random.choice(
                list(probs.keys()), size=22, replace=False, p=list(probs.values())
            )
            simulated_counts.update(selected)
        
        return [int(num) for num, _ in simulated_counts.most_common(n)]
    
    # ==================== HAREKETLİ ORTALAMA ====================
    def moving_average_prediction(self, window: int = 10, n: int = 10) -> List[int]:
        """Hareketli ortalamaya göre tahmin"""
        recent_counts = Counter()
        for idx in range(max(0, len(self.df)-window), len(self.df)):
            recent_counts.update(self.df.iloc[idx][self.number_columns].values)
        
        # En çok çıkanları al
        return [int(num) for num, _ in recent_counts.most_common(n)]
