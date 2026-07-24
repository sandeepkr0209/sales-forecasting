from flask import Flask, request, jsonify, render_template
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
xgb_model = XGBRegressor()
xgb_model.load_model('model/xgboost_model.json')
iso_model  = joblib.load('model/isolation_forest_model.pkl')
scaler     = joblib.load('model/scaler.pkl')

print("All models loaded successfully!")

# ─── PAGE ROUTES ──────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/anomaly-page')
def anomaly_page():
    return render_template('anomaly.html')

# ─── API ROUTES ───────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'Store':          data['store'],
            'Dept':           data['dept'],
            'Year':           data['year'],
            'Month':          data['month'],
            'Week':           data['week'],
            'Quarter':        data['quarter'],
            'IsHoliday':      data['is_holiday'],
            'Temperature':    data['temperature'],
            'Fuel_Price':     data['fuel_price'],
            'CPI':            data['cpi'],
            'Unemployment':   data['unemployment'],
            'Size':           data['size'],
            'Type_Encoded':   data['type_encoded'],
            'Total_MarkDown': data['total_markdown']
        }])
        prediction = np.expm1(xgb_model.predict(input_df)[0])
        return jsonify({
            'status': 'success',
            'predicted_weekly_sales': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/anomaly', methods=['POST'])
def anomaly():
    try:
        data = request.get_json()
        sales = np.array(data['sales']).reshape(-1, 1)
        sales_scaled = scaler.transform(sales)
        predictions  = iso_model.predict(sales_scaled)
        scores       = iso_model.score_samples(sales_scaled)
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                'index':      i,
                'sales':      data['sales'][i],
                'is_anomaly': bool(pred == -1),
                'score':      round(float(score), 4)
            })
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        return jsonify({
            'status':        'success',
            'total_points':  len(results),
            'anomaly_count': anomaly_count,
            'results':       results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'running',
        'models':  ['xgboost', 'isolation_forest'],
        'version': '1.0.0'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)