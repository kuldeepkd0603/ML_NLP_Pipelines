import joblib
import pandas as pd
from utils.constant import regression_constant




def data_preprocessing(data):
    try:
        data['date']=pd.to_datetime(data['date'],dayfirst=True)
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data= data.drop(columns=['date'])
        try:
            encoder = joblib.load(regression_constant.get("encoder"))
        except Exception as e:
            return f"encode not found {e}"
    
        encoded_cat = encoder.transform(data[['WeekStatus','Day_of_week','Load_Type']])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['WeekStatus','Day_of_week','Load_Type']))
         
        df_input = data.drop(columns=['WeekStatus', 'Day_of_week', 'Load_Type'])
        df_input_final = pd.concat([df_input.reset_index(drop=True), encoded_df], axis=1)
         
        try:
            scaler=joblib.load(regression_constant.get("scaler"))
        except Exception as e:
            return f"scaler not found {e}"
        
        X_scaled = scaler.transform(df_input_final)
        return X_scaled
    except Exception as e:
        return f"error occured {e}"
    


def reg_predict(data,ml_algo):
    try:
        if ml_algo=="Linear Regression":
            try:
                lrmodel=joblib.load(regression_constant.get("LR"))
            except Exception as e:
                return f"linear regression model is not found {e}"
            data=data_preprocessing(data)
            result=lrmodel.predict(data)
            return result
        
        
        elif (ml_algo=="Ridge"):
            try:
                rdmodel=joblib.load(regression_constant.get("RLR"))
            except Exception as e:
                return f"ridge regression model is not found {e}"
            data=data_preprocessing(data)
            return rdmodel.predict(data)
        
        
        elif (ml_algo=="Lasso"):
            try:
                llrmodel=joblib.load(regression_constant.get("LLR"))
            except Exception as e:
                return f"lasso regression model is not found {e}"
            data=data_preprocessing(data)
            return llrmodel.predict(data)
        
        elif (ml_algo=="Decision Tree"):
            try:
                dtmodel=joblib.load(regression_constant.get("DTR"))
            except Exception as e:
                return f"decision tree regression model is not found {e}"
            data=data_preprocessing(data)
            return dtmodel.predict(data)
        
        
        elif (ml_algo=="Random Forest"):
            try:
                rfmodel=joblib.load(regression_constant.get("RFR"))
            except Exception as e:
                return f"random forest regression model is not found {e}"
            data=data_preprocessing(data)
            return lrmodel.predict(data)
        
        
        elif (ml_algo=="Gradient Boosting"):
            try:
                gbmodel=joblib.load(regression_constant.get("GBR"))
            except Exception as e:
                return f"gradient boosting regression model is not found {e}"
            data=data_preprocessing(data)
            return gbmodel.predict(data)
        
        
        elif (ml_algo=="Support Vector Regression"):
            try:
                svmodel=joblib.load(regression_constant.get("SVM"))
            except Exception as e:
                return f"support vector regression regression model is not found {e}"
            data=data_preprocessing(data)
            return svmodel.predict(data)
        
        
        elif (ml_algo=="KNN Regressor"):
            try:
                knnmodel=joblib.load(regression_constant.get("KNN"))
            except Exception as e:
                return f"knn regression model is not found {e}"
            data=data_preprocessing(data)
            return knnmodel.predict(data)
        
        
        elif (ml_algo=="AdaBoost Regressor"):
            try:
                adbmodel=joblib.load(regression_constant.get("ABR"))
            except Exception as e:
                return f"adaboost regression model is not found {e}"
            data=data_preprocessing(data)
            return adbmodel.predict(data)
        
        
        else:
            return "wrong model choice"
    except Exception as e:
        return f"error occured {e}"