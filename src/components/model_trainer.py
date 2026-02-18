import os 
import sys 
from dataclasses import dataclass


from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor  
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging 



from src.utils import save_object,evaluate_models


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path =os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                #"CatBoost Regressor": CatBoostRegressor(verbose=False),

            }  

            params = {
                "Random Forest Regressor": {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                
                "Decision Tree Regressor": {
                    'max_depth': [5, 10, 20, 30, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                },
                
                "Gradient Boosting Regressor": {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                },
                
                "KNN Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'euclidean', 'manhattan'],
                    'p': [1, 2]
                },
                
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                
                }
            



            model_report:dict = evaluate_models(X_train=X_train, y_train= y_train, X_test = X_test, y_test=y_test, models= models,params = params) 


            best_model_score = max(model_report.values())


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]       
             
            best_model = models[best_model_name]

            if best_model_score < 0.60 :
                raise CustomException( "No best model found")
            logging.info(f' best found model on both training and testing datasets')
            logging.info(f' the best score among the model is {best_model_score}')
            logging.info(f' the best score among the model is {best_model_name}')


            save_object(

                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model,
            )
        except Exception as e :
            raise CustomException(e,sys)
        
            






