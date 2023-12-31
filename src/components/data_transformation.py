import sys 
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            
            logging.info('Data Transformation initiated')

            fuel = ['Diesel', 'Petrol', 'LPG', 'CNG']
            seller_type = ['Individual', 'Dealer', 'Trustmark Dealer']
            transmission= ['Manual', 'Automatic']
            owner = ['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car']

            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            categorial_columns = ['fuel', 'seller_type', 'transmission', 'owner']
            
            logging.info("Pipeline initiated")

            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
                )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[fuel, seller_type, transmission, owner])),
                    ('scaler', StandardScaler())
                ]
                )

            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline',cat_pipeline,categorial_columns)
            ])
            logging.info("pipeline completed")
            
            print(preprocessor)

            return preprocessor 

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e ,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'selling_price'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("applying preprocessing object on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(

                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('preprocessor pickle file saved')
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
           
        except Exception as e:
            logging.info("Exception ocured in the initite data_transformation")
   
            raise CustomException(e, sys)