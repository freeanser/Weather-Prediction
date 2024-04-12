from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# 載入模型
model = joblib.load('trained_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # 提取特征并转换为NumPy数组
    features = [
        data['Dew_Point_Temp_C'],
        data['Press_kPa'],
        data['Rel_Hum_%'],
        data['Wind_Speed_km/h']
    ]
    data_array = np.array(features).reshape(1, -1)
    # 使用模型进行预测
    prediction = model.predict(data_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
