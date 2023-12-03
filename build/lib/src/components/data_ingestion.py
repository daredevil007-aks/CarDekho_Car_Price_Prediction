from operator import index
import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion methods starts')
        try:
            df = pd.read_csv(os.path.join('Data','cardekho.csv'))
            logging.info('Dataset read as pandas Dataframe')
            df= df.drop('torque',axis=1)
            df['max_power'] = df['max_power'].str.replace(' bhp','')
            df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
            df['engine']= df['engine'].str.replace(' CC','')
            df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
            df['mileage'] = df['mileage'].str.replace(' kmpl','')
            df['mileage'] = df['mileage'].str.replace(' km/kg','')
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
            df = df.dropna()

            df = df.drop(['name'], axis=1)
            logging.info('EDA Completed performing ingestion')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train test split")

            train_set, test_set = train_test_split(df, test_size=20)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
            
        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e, sys)


