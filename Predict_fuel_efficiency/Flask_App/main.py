import pickle
from flask import Flask, request, jsonify
from Model_Files.ml_model import predict_mpg

app = Flask("mpg_prediction")

@app.route('/', methods=['POST'])
def predict():
    vehicle_config = request.get_json()

    with open('./Model_Files/model.bin','rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_mpg(vehicle_config, model)

    response = {
        'mpg_predictions' : list(predictions)
    }
    return jsonify(response)

# @app.route('/', methods=['GET'])
# def ping():
#     return "Hello Flask App!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)