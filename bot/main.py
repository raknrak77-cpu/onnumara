import argparse
from prediction_engine import PredictionEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest', 'forward', 'all'], default='all')
    args = parser.parse_args()
    
    engine = PredictionEngine(excel_path="onnumara_2020.xlsx")
    engine.load_and_clean()
    
    print(f"\n📊 Veri setinde {len(engine.df)} çekiliş var")
    
    if args.mode in ['backtest', 'all']:
        print("\n🔙 Backtest yapılıyor...")
        backtest_results = engine.run_backtest(train_size=500, test_size=50)
        
        # Başarı istatistikleri
        scores = [r['basarisayisi'] for r in backtest_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"📈 Ortalama doğru tahmin: {avg_score:.2f} / 10")
        
        engine.save_results(backtest_results, "backtest.json")
        
    if args.mode in ['forward', 'all']:
        print("\n🔮 İleri tahmin yapılıyor...")
        forward_results = engine.predict_future(n_predictions=3)
        engine.save_results(forward_results, "forward_predictions.json")
        
        print("\n🔮 Tahminler:")
        for pred in forward_results:
            print(f"  {pred['tahmin_no']}. {pred['tarih_tahmini']}:")
            print(f"     Ensemble: {pred['ensemble_top10'][:5]}...")

if __name__ == "__main__":
    main()
