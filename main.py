# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:04:03 2021

@author: YG
"""

import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Disable warning PyplotGlobalUseWarning
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def load_data(display):
    return shap.datasets.boston(display=display)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.set_page_config(page_title="XAI with SHAP")

st.title("XAI with SHAP")

# Sidebar
# Input sidebar subheader
st.sidebar.header('Options')
st.sidebar.subheader("Show plots:")
show_dataset_dictionary = st.sidebar.checkbox(label='Dataset dictionary ', value=False)
show_force_plots = st.sidebar.checkbox(label='Force plots ', value=True)
show_feature_importance_plot = st.sidebar.checkbox(label='Feature importance ', value=True)
show_mean_importance_plot = st.sidebar.checkbox(label='Mean importance', value=True)
show_dependence_plot = st.sidebar.checkbox(label='Dependence plot', value=True)

# Load dataset
X, y = load_data(display=False)
X_display, y_display = load_data(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

# train XGBoost model
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

# Explain dataset's features (dataset dictionary)
# TODO poner mas features
dataset_dictionary = pd.DataFrame({
    'Feature': ["CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "...more"],
    'Description': ["per capita crime rate by town",
                    "proportion of residential land zoned for lots over 25,000 sq.ft.",
                    "proportion of non-retail business acres per town",
                    "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                    "nitric oxides concentration (parts per 10 million)",
                    "...more"]
})

if show_dataset_dictionary:
    st.write("Dataset dictionary:")
    st.write(dataset_dictionary)
    st.write("Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset")

if show_force_plots:
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    selected_index = st.selectbox('Index', (list(range(10))))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index, :], X.iloc[selected_index, :]))

    # visualize the training set predictions
    selected_num = st.select_slider('number of examples', options=list(np.arange(10, 100, 10)))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:selected_num, :], X.iloc[:selected_num, :]), 400)

# visualize the summary plot
# st.pyplot(shap.summary_plot(shap_values, X)) another way to show the summary plot
if show_feature_importance_plot:
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot()

# visualize Bar chart of mean importance
if show_mean_importance_plot:
    plt.title('Bar chart of mean importance')
    shap.summary_plot(shap_values, X_display, plot_type="bar")
    st.pyplot()

# visualize SHAP for each feature
# TODO cambiar por los otros graficos, al final de la documentacion esta
if show_dependence_plot:
    st.title("SHAP Dependence Plots")
    for name in X_train.columns:
        shap.dependence_plot(name, shap_values, X, display_features=X_display)
        st.pyplot()
