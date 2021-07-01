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

    if dataset_to_show == 'Boston housing':
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
    'datasets_longname': ['Boston housing', 'Adult census', 'Nhanes I', 'Communities and crime']
}

# Sidebar
st.sidebar.header('Options')
selected_dataset = st.sidebar.selectbox("Dataset", datasets['datasets_longname'], index=0)
clf = st.sidebar.selectbox('Choose Classifier:',['XGBoost','Random Forest','Decision Tree'])
st.sidebar.subheader("Show plots:")
show_force_plots = st.sidebar.checkbox(label='Force plots ', value=True)
show_feature_importance_plot = st.sidebar.checkbox(label='Feature importance ', value=True)
show_mean_importance_plot = st.sidebar.checkbox(label='Mean importance', value=True)
show_dependence_plot = st.sidebar.checkbox(label='Dependence plot', value=True)


# Load dataset
X, y = load_data(display=False, dataset_to_show=selected_dataset)
X_display, y_display = load_data(display=True, dataset_to_show=selected_dataset)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

##---------------XGBoost
if clf == 'XGBoost':
     
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    # train XGBoost model
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100) 

##-----------------RF
if clf == 'Random Forest':

    ## Random Foreast
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

##------------------Text
if clf == 'Decision Tree':
    
    ## Decision Tree
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state = 0)
    model.fit(X_train, y_train)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

# Explain dataset's features (dataset dictionary)
datasets_links = {
    'Boston housing': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset',
    'Adult census': 'https://archive.ics.uci.edu/ml/datasets/adult',
    'Nhanes I': 'https://wwwn.cdc.gov/nchs/nhanes/nhanes1/',
    'Communities and crime': 'https://github.com/slundberg/shap/blob/master/data/NHANESI_X.csv'
}

st.write("Dataset information: %s" % datasets_links[selected_dataset])

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
if show_force_plots:
    st.header("Force plots")
    st.subheader("Visualize a single prediction")
    selected_index = st.number_input('Prediction index (max. %s)' % (len(shap_values)-1), value=0,min_value=0,
                                     max_value=(len(shap_values)-1), step=1)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index, :], X.iloc[selected_index, :]))

    # visualize the training set predictions
    st.subheader("Visualize many predictions")
    selected_num = st.select_slider('Number of examples', options=list(np.arange(10, (len(shap_values)-1), 10)))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:selected_num, :], X.iloc[:selected_num, :]), 400)

    with st.beta_expander("More about these plots"):
        st.markdown("""
           The Force plots are effective at showing how the model arrived at its decision.
           
           The SHAP values displayed are additive. Once the negative values (blue) are substracted from the positive 
           values (red), the distance from the base value to the output remains.
        """)

# visualize the summary plot
if show_feature_importance_plot:
    st.header("SHAP Summary Plot")
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot()

    with st.beta_expander("More about this plot"):
        st.markdown("""
           This summary plot (beeswarm plot) is designed to display an information-dense summary of how the top features
            in a dataset impact the model’s output. Each instance the given explanation is represented by a single dot 
            on each feature fow. The x position of the dot is determined by the SHAP value of that feature, and dots 
            “pile up” along each feature row to show density. 
            
            Color is used to display the original value of a feature. 
        """)

# visualize Bar chart of mean importance
if show_mean_importance_plot:
    st.header('Bar chart of mean importance')
    shap.summary_plot(shap_values, X_display, plot_type="bar")
    st.pyplot()

    with st.beta_expander("More about this plot"):
        st.markdown("""
            This is a global feature importance plot, where the global importance of each feature is taken to be the 
            mean absolute value for that feature over all the given samples.
        """)

# visualize SHAP Dependence Plots
if show_dependence_plot:
    st.header("SHAP Dependence Plots")
    column1, column2 = st.beta_columns(2)
    feature_dependence_plot = column1.selectbox('Feature', X_train.columns, index=0)
    interaction_selector = column2.selectbox('Interaction feature', X_train.columns, index=5)
    shap.dependence_plot(feature_dependence_plot, shap_values, X,  interaction_index=interaction_selector)
    st.pyplot()

    with st.beta_expander("More about this plot"):
        st.markdown("""
            The dependence plot is a scatter plot that shows the effect a single feature has on the predictions made by 
            the model. 

            - Each dot is a single prediction (row) from the dataset.
            - The x-axis is the value of the feature (from the X matrix).
            - The y-axis is the SHAP value for that feature, which represents how much knowing that feature's value 
               changes the output of the model for that sample's prediction. 
            - The color corresponds to a second feature that may have an interaction effect with the feature we are 
               plotting. If an interaction effect is present between this other feature and the feature we are plotting 
               it will show up as a distinct vertical pattern of coloring.

        """)
