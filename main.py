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
def load_data(display, dataset_to_show):
    dataset = None

    if dataset_to_show == 'Boston housing data':
        dataset = shap.datasets.boston(display=display)
    elif dataset_to_show == 'Adult census':
        dataset = shap.datasets.adult(display=display)
    elif dataset_to_show == 'Nhanes I':
        dataset = shap.datasets.nhanesi(display=display)
    elif dataset_to_show == 'Communities and crime':
        dataset = shap.datasets.communitiesandcrime(display=display)
    return dataset


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Webapp title
st.set_page_config(page_title='XAI with SHAP', layout="wide")

st.title("Explainable Artificial Intelligence with SHAP")

# Datasets
datasets = {
    'datasets_longname': ['Boston housing data', 'Adult census', 'Nhanes I', 'Communities and crime']
}

# Sidebar
st.sidebar.header('Options')
selected_dataset = st.sidebar.selectbox("Dataset", datasets['datasets_longname'], index=0)
st.sidebar.subheader("Show plots:")
show_dataset_dictionary = st.sidebar.checkbox(label='Dataset dictionary ', value=False)
show_force_plots = st.sidebar.checkbox(label='Force plots ', value=True)
show_feature_importance_plot = st.sidebar.checkbox(label='Feature importance ', value=True)
show_mean_importance_plot = st.sidebar.checkbox(label='Mean importance', value=True)
show_dependence_plot = st.sidebar.checkbox(label='Dependence plot', value=True)


# Load dataset
X, y = load_data(display=False, dataset_to_show=selected_dataset)
X_display, y_display = load_data(display=True, dataset_to_show=selected_dataset)

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
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV"],
    'Description': ["per capita crime rate by town",
                    "proportion of residential land zoned for lots over 25,000 sq.ft.",
                    "proportion of non-retail business acres per town",
                    "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                    "nitric oxides concentration (parts per 10 million)",
                    "average number of rooms per dwelling",
                    "proportion of owner-occupied units built prior to 1940",
                    "weighted distances to five Boston employment centres",
                    "index of accessibility to radial highways",
                    "full-value property-tax rate per $10,000",
                    "pupil-teacher ratio by town",
                    "1000(Bk - 0.63)^2 where Bk is the proportion of black people by town",
                    "% lower status of the population",
                    "Median value of owner-occupied homes in $1000’s"]
})
if show_dataset_dictionary:
    st.write("Dataset dictionary:")
    st.table(dataset_dictionary)
    st.write("Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset")

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
if show_force_plots:
    st.header("Visualize a single prediction")
    selected_index = st.number_input('Prediction index (max. %s)' % (len(shap_values)-1), value=0,min_value=0,
                                     max_value=(len(shap_values)-1), step=1)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index, :], X.iloc[selected_index, :]))

    # visualize the training set predictions
    st.header("Visualize many predictions")
    selected_num = st.select_slider('Number of examples', options=list(np.arange(10, (len(shap_values)-1), 10)))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:selected_num, :], X.iloc[:selected_num, :]), 400)

# visualize the summary plot
if show_feature_importance_plot:
    st.header("SHAP Summary Plot")
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot()

# visualize Bar chart of mean importance
if show_mean_importance_plot:
    st.header('Bar chart of mean importance')
    shap.summary_plot(shap_values, X_display, plot_type="bar")
    st.pyplot()

# visualize SHAP Dependence Plots
if show_dependence_plot:
    st.header("SHAP Dependence Plots")
    column1, column2 = st.beta_columns(2)
    feature_dependence_plot = column1.selectbox('Feature', X_train.columns, index=0)
    interaction_selector = column2.selectbox('Interaction feature', X_train.columns, index=5)
    shap.dependence_plot(feature_dependence_plot, shap_values, X,  interaction_index=interaction_selector)
    st.pyplot()
