import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, mae, mse, rmse, mape

class CART:
    def __init__(self, task_type, max_depth=None, min_samples_split=2):
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or len(X) <= self.min_samples_split:
            if self.task_type == 'regression':
                leaf_value = np.mean(y)
                return {'leaf': True, 'value': leaf_value}
            elif self.task_type == 'classification':
                class_counts = np.bincount(y)
                class_probability = class_counts / np.sum(class_counts)
                return {'leaf': True, 'class': np.argmax(class_counts), 'probability': class_probability}

        best_feature, best_threshold = self._find_best_split(X, y)
        left_child_indices = X[:, best_feature] <= best_threshold
        right_child_indices = X[:, best_feature] > best_threshold

        node = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_child_indices], y[left_child_indices], depth + 1),
            'right': self._build_tree(X[right_child_indices], y[right_child_indices], depth + 1)
        }
        return node

    def _find_best_split(self, X, y):
        if self.task_type == 'regression':
            best_mse = float('inf')
            best_feature = None
            best_threshold = None

            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    left_indices = X[:, feature] <= threshold
                    right_indices = X[:, feature] > threshold

                    mse = self._mse(y[left_indices], y[right_indices])

                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold

            return best_feature, best_threshold
        elif self.task_type == 'classification':
            best_gini = float('inf')
            best_feature = None
            best_threshold = None

            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    left_indices = X[:, feature] <= threshold
                    right_indices = X[:, feature] > threshold

                    gini = self._gini(y[left_indices], y[right_indices])

                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold

            return best_feature, best_threshold

    def _mse(self, left_y, right_y):
        left_mse = np.mean((left_y - np.mean(left_y))**2)
        right_mse = np.mean((right_y - np.mean(right_y))**2)
        mse = (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))
        return mse

    def _gini(self, left_y, right_y):
        left_gini = 1 - np.sum((np.bincount(left_y) / len(left_y))**2)
        right_gini = 1 - np.sum((np.bincount(right_y) / len(right_y))**2)
        gini = (len(left_y) * left_gini + len(right_y) * right_gini) / (len(left_y) + len(right_y))
        return gini

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        if X.ndim == 1:
            X = np.expand_dims(X, 0)

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if 'leaf' in node:
            if self.task_type == 'regression':
                return node['value']
            elif self.task_type == 'classification':
                return node['class']

        feature_value = x[node['feature']]
        
        if not isinstance(feature_value, (int, float)):
            return self._traverse_tree(x, node['left'])
        
        if feature_value <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

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
        
        st.title("Обучение модели sclearn") 
        progress_bar = st.progress(0)
        i = 0
        regressor = DecisionTreeRegressor()
        param_dist = {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "max_depth": randint(1, 4),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10)
        }
        progress_bar.progress(i + 25)
        i = i + 25
        reg_random_search = RandomizedSearchCV(regressor, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
        progress_bar.progress(i + 25)
        i = i + 25
        reg_random_search.fit(X_train, y_train)
        progress_bar.progress(i + 25)
        i = i + 25
        best_regressor = reg_random_search.best_estimator_
        progress_bar.progress(i + 25)
        i = i + 25
        st.title("Метрики модели sclearn")
        metrics_regressor = calculate_metrics(best_regressor, X_test, y_test)
        st.write(f'MAE: {metrics_regressor[1]}')
        st.write(f'MSE: {metrics_regressor[2]}')
        st.write(f'RMSE: {metrics_regressor[3]}')
        st.write(f'MAPE: {metrics_regressor[4]}')
        st.write(f'R^2: {metrics_regressor[0]}')
        
        st.title("Обучение кастомного CART")
        progress_bar = st.progress(0)
        i = 0
        regressor = CART(task_type='regression', max_depth=5, min_samples_split=2)
        progress_bar.progress(i + 50)
        i = i + 50
        regressor.fit(X_train, y_train)
        progress_bar.progress(i + 50)
        i = i + 50
        
        st.title("Метрики кастомного CART")
        metrics_regressor = calculate_metrics(regressor, X_test, y_test)
        st.write(f'MAE: {metrics_regressor[1]}')
        st.write(f'MSE: {metrics_regressor[2]}')
        st.write(f'RMSE: {metrics_regressor[3]}')
        st.write(f'MAPE: {metrics_regressor[4]}')
        st.write(f'R^2: {metrics_regressor[0]}')
        
        
        



if uploaded_file is not None and dataset == "Классификация":
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t')

    button_clicked = st.button("Начать обучение")       
    
    if button_clicked:
        st.title("Предобработка данных")   
        df = df.drop(['pizza_ingredients', 'order_date', 'order_time', 'pizza_name_id', 'order_id', 'pizza_id'], axis=1)
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
        X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Предобработка! Завершена")
        
        st.title("Обучение модели sclearn") 
        progress_bar = st.progress(0)
        i = 0
        classifier = DecisionTreeClassifier()
        param_dist = {
            "criterion": ["gini", "entropy"],
            "max_depth": randint(1, 4),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10)
        }
        progress_bar.progress(i + 25)
        i = i + 25
        clf_random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
        progress_bar.progress(i + 25)
        i = i + 25
        clf_random_search.fit(X_train, y_train)
        progress_bar.progress(i + 25)
        i = i + 25
        best_classifier = clf_random_search.best_estimator_
        progress_bar.progress(i + 25)
        i = i + 25
        
        st.title("Метрики модели sclearn")
        clf_predictions = best_classifier.predict(X_test)
        st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, clf_predictions)))
        st.write('Precision: {:.3f}'.format(precision_score(y_test, clf_predictions, average='weighted')))
        st.write('Recall: {:.3f}'.format(recall_score(y_test, clf_predictions, average='weighted')))
        st.write('F1-score: {:.3f}'.format(f1_score(y_test, clf_predictions, average='weighted')))
        
        st.title("Обучение кастомного CART")
        progress_bar = st.progress(0)
        i = 0
        classifier = CART(task_type='classification', max_depth=10, min_samples_split=5)
        progress_bar.progress(i + 50)
        i = i + 50
        classifier.fit(X_train, y_train)
        progress_bar.progress(i + 50)
        i = i + 50
        
        st.title("Метрики кастомного CART")
        clf_predictions = classifier.predict(X_test)
        st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, clf_predictions)))
        st.write('Precision: {:.3f}'.format(precision_score(y_test, clf_predictions, average='weighted')))
        st.write('Recall: {:.3f}'.format(recall_score(y_test, clf_predictions, average='weighted')))
        st.write('F1-score: {:.3f}'.format(f1_score(y_test, clf_predictions, average='weighted')))