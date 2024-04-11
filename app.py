from flask import Flask, request, jsonify
import numpy as np  # 添加此行以转换数据点为数组
import joblib

app = Flask(__name__)

# 載入模型
model = joblib.load('trained_model.joblib')

# 定義預測端點
@app.route('/predict', methods=['POST'])
def predict():
    # 從請求中獲取資料
    data = request.get_json()
    # 將資料轉換為 NumPy 數組
    data_array = np.array(data['features'])  # 假設 'features' 是要預測的特徵
    # 使用模型進行預測
    prediction = model.predict(data_array.reshape(1, -1))  # 這裡假設您的模型僅接受一個樣本，所以使用 reshape 進行轉換
    # 回傳預測結果
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
