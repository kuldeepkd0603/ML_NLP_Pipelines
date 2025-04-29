regression_constant={
    "encoder":"models/regression/onehotencoder.pkl",
    "scaler":"models/regression/standardscaler.pkl",
    "LR":"models/regression/linear_regression_model.pkl",
    "RLR":"models/regression/ridge_model.pkl",
    "LLR":"models/regression/lasso_model.pkl",
    "DTR":"models/regression/decision_tree_model.pkl",
    "RFR":"models/regression/random_forest_model.pkl",
    "GBR":"models/regression/gradient_boosting_model.pkl",
    "SVM":"models/regression/support_vector_regression_model.pkl",
    "KNN":"models/regression/knn_regressor_model.pkl",
    "ABR":"models/regression/adaboost_regressor_model.pkl"
}  


classification_constant={
    "scaler":"models/classification/_classi_scaler.pkl",
    "LRC":"models/classification/logistic_regression_model.pkl",
    "NBC":" models/classification/naive_bayes_model.pkl",
    "DTC":"models/classification/decision_tree_model.pkl",
    "RFC":"models/classification/random_forest_model.pkl",
    "GBC":"models/classification/gradient_boost_model.pkl",
    "SVMC":"models/classification/svm_model.pkl",
    "KNNC":"models/classification/knn_model.pkl",
    "ABC":"models/classification/adaboost_model.pkl"
    
}

nlp_constants={
    "tf-idf-scaler":"models/nlp/Tf-Idf/scaler.pkl",
    "tf-idf-vector":"models/nlp//Tf-Idf/vectorizer.pkl",
    "tf-idf-model":"models/nlp/Tf-Idf/logistic_regression_model.pkl",
    "word2vec-vectorizer":"models/nlp/word2vec/word2vec.model",
    "word2vec-ml-model":"models/nlp/word2vec/models_random_forest_model.pkl"
}