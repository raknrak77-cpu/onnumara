#!/usr/bin/env python3
"""
On Numara 22'lik Set Tahmin Botu
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.predictor_22 import Predictor22

def main():
    print("\n" + "=" * 60)
    print("🎯 ON NUMARA 22'LİK SET TAHMİN BOTU")
    print("   80 sayı içinden en güçlü 22 sayıyı önerir")
    print("=" * 60 + "\n")
    
    bot = Predictor22()
    bot.load_data()
    
    # Raporu oluştur ve yazdır
    report = bot.generate_report()
    print(report)
    
    # Sonuçları kaydet
    ensemble = bot.ensemble_22_model()
    bot.save_results(ensemble, "predictions_22.json")
    
    # Ayrıca raporu da kaydet
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/report_22.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n💾 Kaydedildi: outputs/report_22.txt")

if __name__ == "__main__":
    main()
