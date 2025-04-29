import pandas as pd
import joblib
import numpy as np
from utils.constant import classification_constant

def clssi_preprocess(data):
    try:
        if data.at[0, 'four_g'] == 1:
            data.at[0, 'three_g'] = 1
            data.at[0, 'touch_screen'] = 1
            data.at[0, 'wifi'] = 1
            data.at[0, 'blue'] = 1
          
        try:
            scaler=joblib.load(classification_constant.get("scaler"))
        except Exception as e:
            return f"scaler not found {e}"
        
        X_scaled = scaler.transform(data)
        return X_scaled
    except Exception as e:
        return f"error occured {e}"

def clssi_predict(ml_algo,data):
    try:
        
        if ml_algo=="Logistic Regression":
            try:
                lrmodel=joblib.load(classification_constant.get("LRC"))
            except Exception as e:
                return f"Logistic Regression model is not found {e}"
            data=clssi_preprocess(data)
            result=lrmodel.predict(data)
            return result
        
        
        elif (ml_algo=="Decision Tree"):
            try:
                rdmodel=joblib.load(classification_constant.get("DTC"))
            except Exception as e:
                return f"Decision Tree model is not found {e}"
            data=clssi_preprocess(data)
            return rdmodel.predict(data)
        
        
        elif (ml_algo=="Random Forest"):
            try:
                llrmodel=joblib.load(classification_constant.get("RFC"))
            except Exception as e:
                return f"Random Forest model is not found {e}"
            data=clssi_preprocess(data)
            return llrmodel.predict(data)
        
        elif (ml_algo=="SVM"):
            try:
                dtmodel=joblib.load(classification_constant.get("SVMC"))
            except Exception as e:
                return f"SVM regression model is not found {e}"
            data=clssi_preprocess(data)
            return dtmodel.predict(data)
        
        
        elif (ml_algo=="Gradient Boost"):
            try:
                rfmodel=joblib.load(classification_constant.get("GBC"))
            except Exception as e:
                return f"Gradient Boost regression model is not found {e}"
            data=clssi_preprocess(data)
            return lrmodel.predict(data)
        
        
        elif (ml_algo=="AdaBoost"):
            try:
                gbmodel=joblib.load(classification_constant.get("ABC"))
            except Exception as e:
                return f"AdaBoost regression model is not found {e}"
            data=clssi_preprocess(data)
            return gbmodel.predict(data)
        
        
        elif (ml_algo=="Naive Bayes"):
            try:
                svmodel=joblib.load(classification_constant.get("NBC"))
            except Exception as e:
                return f"Naive Bayes regression model is not found {e}"
            data=clssi_preprocess(data)
            return svmodel.predict(data)
        
        
        elif (ml_algo=="KNN "):
            try:
                knnmodel=joblib.load(classification_constant.get("KNNC"))
            except Exception as e:
                return f"knn classifier model is not found {e}"
            data=clssi_preprocess(data)
            return knnmodel.predict(data)
        
        else:
            return "wrong model choice"
    except Exception as e:
        return f"error occured {e}"