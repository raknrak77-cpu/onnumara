import argparse
from prediction_engine import PredictionEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest', 'forward', 'all'], default='all')
    args = parser.parse_args()
    
    print("🚀 On Numara Prediction Bot Başlıyor...")
    
    engine = PredictionEngine(excel_path="onnumara_2020.xlsx")
    engine.load_and_clean()
    
    print(f"\n📊 Veri setinde {len(engine.df)} çekiliş var")
    print(f"🔢 Sayı sütunları: {len(engine.number_columns)} adet")
    
    if args.mode in ['backtest', 'all']:
        print("\n🔙 Backtest yapılıyor...")
        backtest_results = engine.run_backtest(train_size=500, test_size=50)
        
        if backtest_results:
            scores = [r['basarisayisi'] for r in backtest_results]
            avg_score = sum(scores) / len(scores)
            print(f"📈 Ortalama doğru tahmin: {avg_score:.2f} / 10")
            print(f"📊 En iyi: {max(scores)} doğru, En kötü: {min(scores)} doğru")
        else:
            print("⚠️ Backtest sonuç üretemedi!")
            
        engine.save_results(backtest_results, "backtest.json")
        
    if args.mode in ['forward', 'all']:
        print("\n🔮 İleri tahmin yapılıyor...")
        forward_results = engine.predict_future(n_predictions=3)
        
        print("\n🔮 Tahminler:")
        for pred in forward_results:
            print(f"\n  📅 {pred['tahmin_no']}. Tahmin - {pred['tarih_tahmini']}:")
            print(f"     🎯 Ensemble (önerilen): {pred['ensemble_top10'][:10]}")
        
        engine.save_results(forward_results, "forward_predictions.json")
    
    print("\n✅ Bot tamamlandı!")

if __name__ == "__main__":
    main()
