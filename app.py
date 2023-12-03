from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')

    else:
        data=CustomData(
            year = float(request.form.get('year')),
            km_driven= float(request.form.get('km_driven')),
            mileage= float(request.form.get('mileage')),
            engine= float(request.form.get('engine')),
            max_power= float(request.form.get('max_power')),
            seats = float(request.form.get('seats')),
            fuel= request.form.get('fuel'),
            seller_type= request.form.get('seller_type'),
            transmission= request.form.get('transmission'),
            owner= request.form.get('owner')

        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)
        print(results)

        return render_template('result.html', final_result=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0")