import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
datasets = ['Регрессия', 'Классификация']

dataset = st.selectbox("Выберите тип датасета", datasets)

uploaded_file = st.file_uploader("Выберите файл датасета", type=["csv"])

if uploaded_file is not None and dataset == "Регрессия":
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t') 

    st.write("Загруженный датасет:", df)

    st.title("Датасет Moldova Сars")

    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))
    selected_cols = ['Year','Distance','Engine_capacity(cm3)','Price(euro)']
    selected_df = df[selected_cols]
    sns.heatmap(selected_df.corr().round(2), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)
    
    st.header("Гистограммы для основных признаков")
    
    columns = ['Year','Distance','Engine_capacity(cm3)','Price(euro)']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)
    
    st.header("Ящик с усами для основных признаков")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)
   
    
    columns = ['Fuel_type', 'Transmission']
    st.header("Круговая диаграмма основных категориальных признаков")
    for col in columns:
        plt.figure(figsize=(8, 8))
        df[col].value_counts().plot.pie(autopct='')
        plt.title(f'{col}')
        plt.ylabel('')
        st.pyplot(plt)
        
        
if uploaded_file is not None and dataset == "Классификация":
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t') 
    
    
    df['pizza_size'] = df['pizza_size'].replace({'S': 'Small', 'M': 'Medium', 'L': 'Large', 'XL' : 'X-Large' , 'XXL' : 'XX-Large'}, inplace=False)
   
    st.write("Загруженный датасет:", df)

    st.title("Датасет Pizza Sales")

    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))
    selected_cols = ['unit_price','total_price', 'quantity']
    selected_df = df[selected_cols]
    sns.heatmap(selected_df.corr().round(2), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)
    
    st.header("Гистограммы для основных признаков")
    
    columns = ['unit_price','total_price']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)
    
    st.header("Ящик с усами для основных признаков")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)
    
    columns = ['pizza_category','pizza_size']
    st.header("Круговая диаграмма основных категориальных признаков")
    for col in columns:
        plt.figure(figsize=(8, 8))
        df[col].value_counts().plot.pie(autopct='')
        plt.title(f'{col}')
        plt.ylabel('')
        st.pyplot(plt)
        