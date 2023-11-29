from json import load
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 year:float,
                 km_driven:float,
                 mileage:float,
                 engine:float,
                 max_power:float,
                 seats:float,
                 fuel:str,
                 seller_type:str,
                 transmission:str,
                 owner:str):
        