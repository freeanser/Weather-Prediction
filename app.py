from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 載入模型
model = joblib.load('trained_model.joblib')

# 定義預測端點
@app.route('/predict', methods=['POST'])
def predict():
    # 從請求中獲取資料
    data = request.get_json()
    # 使用模型進行預測
    prediction = model.predict(data)
    # 回傳預測結果
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
