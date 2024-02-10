import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import tensorflow as tf
import pickle
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier


datasets = ['Регрессия', 'Классификация']

dataset = st.selectbox("Выберите тип датасета", datasets)

uploaded_file = st.file_uploader("Выберите файл датасета", type=["csv"])

if uploaded_file is not None and dataset == "Регрессия":
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t')

    button_clicked = st.button("Начать обучение")  

    if button_clicked:
        st.title("Предобработка данных")   
    
        progress_bar = st.progress(0)
        i = 0
        progress_bar.progress(i + 10)
        i = i + 10
        cols = df.columns.tolist()
        cols = [col.lower().replace(" ", "_") for col in cols]
        df.columns = cols
        progress_bar.progress(i + 10)
        i = i + 10
        if df.isnull().any().any():
            for column in df.columns:
                if df[column].dtype == 'int64':
                    df[column].fillna(df[column].median(), inplace=True)
                elif df[column].dtype == 'float64':
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode().iloc[0], inplace=True)
        progress_bar.progress(i + 10)
        i = i + 10
        df = df.drop_duplicates().reset_index(drop=True)
        progress_bar.progress(i + 10)
        i = i + 10
        for column in df.columns:
            if df[column].dtype == float:
                df[column] = df[column].astype(int)

        df = pd.get_dummies(df)
        
        outlier = df[['year', 'distance', 'engine_capacity(cm3)', 'price(euro)']]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3-Q1
        df_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(df_filtered.index.values)
        df_filtered = df[df.index.isin(index_list)]
        progress_bar.progress(i + 20)
        i = i + 20
        
        scaler = StandardScaler()

        numeric_features = ['year', 'distance', 'engine_capacity(cm3)', 'price(euro)'] 
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        progress_bar.progress(i + 20)
        i = i + 20
        
        y = df_filtered["price(euro)"]
        X = df_filtered.drop(["price(euro)"], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Предобработка! Завершена")
        
        param_grid = {
            'alpha': [0.1, 1.0, 10.0]
        }
        st.title("Обучение модели") 
        progress_bar = st.progress(0)
        i = 0
        progress_bar.progress(i + 20)
        i = i + 20
        model = Ridge()
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        progress_bar.progress(i + 30)
        i = i + 30
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        progress_bar.progress(i + 30)
        i = i + 30
        y_pred = model.predict(X_test)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Модель обучена!")
        
        st.title("Метрики модели")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        st.write('R2: {:.3f}'.format(r2))
        st.write('MAE: {:.3f}'.format(mae))
        st.write('MSE: {:.3f}'.format(mse))
        st.write('RMSE: {:.3f}'.format(rmse))
        st.write('MAPE: {:.3f}'.format(mape))



if uploaded_file is not None and dataset == "Классификация":
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t')

    button_clicked = st.button("Начать обучение")       
    
    if button_clicked:
        st.title("Предобработка данных")   
        df.drop(['pizza_ingredients', 'order_date', 'order_time', 'pizza_name_id', 'order_id', 'pizza_id', 'pizza_name'], axis=1)
        progress_bar = st.progress(0)
        i = 0
        progress_bar.progress(i + 10)
        i = i + 10
        cols = df.columns.tolist()
        cols = [col.lower().replace(" ", "_") for col in cols]
        df.columns = cols
        progress_bar.progress(i + 10)
        i = i + 10
        if df.isnull().any().any():
            for column in df.columns:
                if df[column].dtype == 'int64':
                    df[column].fillna(df[column].median(), inplace=True)
                elif df[column].dtype == 'float64':
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode().iloc[0], inplace=True)
        progress_bar.progress(i + 10)
        i = i + 10
        df = df.drop_duplicates().reset_index(drop=True)
        progress_bar.progress(i + 10)
        i = i + 10
        for column in df.columns:
            if df[column].dtype == float:
                df[column] = df[column].astype(int)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        df["pizza_category"] = le.fit_transform(df['pizza_category'])
        df = pd.get_dummies(df)
        
        outlier = df[['unit_price', 'total_price']]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3-Q1
        df_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(df_filtered.index.values)
        df_filtered = df[df.index.isin(index_list)]
        progress_bar.progress(i + 20)
        i = i + 20
        
        progress_bar.progress(i + 20)
        i = i + 20
        y = df_filtered["pizza_category"]
        X = df_filtered.drop(["pizza_category"], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Предобработка! Завершена")
        
        param_grid = {
            'alpha': [0.1, 1.0, 10.0]
        }
        st.title("Обучение модели") 
        progress_bar = st.progress(0)
        i = 0
        progress_bar.progress(i + 20)
        i = i + 20
        model = RidgeClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        progress_bar.progress(i + 30)
        i = i + 30
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        progress_bar.progress(i + 30)
        i = i + 30
        y_pred = model.predict(X_test)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Модель обучена!")
        
        st.title("Метрики модели")
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
        st.write('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
        st.write('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)))
        st.write('ROC-AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)))