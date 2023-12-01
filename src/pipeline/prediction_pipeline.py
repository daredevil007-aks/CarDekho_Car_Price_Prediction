from json import load
import os
import sys
from turtle import distance
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
        self.year = year
        self.km_driven = km_driven
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats
        self.fuel = fuel
        self.seller_type = seller_type
        self.transmission = transmission
        self.owner = owner

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'year':[self.year],
                'km_driven':[self.km_driven],
                'mileage':[self.mileage],
                'engine':[self.engine],
                'max_power':[self.max_power],
                'seats':[self.seats],
                'fuel':[self.fuel],
                'seller_type':[self.seller_type],
                'transmission':[self.transmission],
                'owner':[self.owner]
            }        
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)