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

# Webapp title
st.set_page_config(page_title='XAI with SHAP')

st.title("Explainable Artificial Intelligence with SHAP")

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
st.write("Dataset dictionary:")
st.write(dataset_dictionary)
st.write("Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset")

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st.header("Visualize a single prediction")
selected_index = st.number_input('Prediction index (max. %s)' % (len(shap_values)-1), value=0,min_value=0,
                                 max_value=(len(shap_values)-1), step=1)
st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index, :], X.iloc[selected_index, :]))

# visualize the training set predictions
st.header("Visualize many predictions")
selected_num = st.select_slider('Number of examples', options=list(np.arange(10, (len(shap_values)-1), 10)))
st_shap(shap.force_plot(explainer.expected_value, shap_values[:selected_num, :], X.iloc[:selected_num, :]), 400)

# visualize the summary plot
st.header("SHAP Summary Plot")
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X, show=False)
st.pyplot()

# visualize Bar chart of mean importance
st.header('Bar chart of mean importance')
shap.summary_plot(shap_values, X_display, plot_type="bar")
st.pyplot()

# visualize SHAP Dependence Plots
st.header("SHAP Dependence Plots")
column1, column2 = st.beta_columns(2)
feature_dependence_plot = column1.selectbox('Feature', X_train.columns, index=0)
interaction_selector = column2.selectbox('Interaction feature', X_train.columns, index=5)
shap.dependence_plot(feature_dependence_plot, shap_values, X,  interaction_index=interaction_selector)
st.pyplot()
