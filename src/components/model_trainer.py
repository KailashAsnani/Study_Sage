import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_reports:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_reports.values()))

            best_model_name = list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model.")
            
            logging.info(f"Best model on training and testing datatsets.")


            save_object(file_path=self.model_trainer_config.model_trainer_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_train)
            r2_scqure = r2_score(y_train,predicted)

            return r2_scqure


        except Exception as e:
            raise CustomException(e,sys) 