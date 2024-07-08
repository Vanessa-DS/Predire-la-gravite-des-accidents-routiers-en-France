import streamlit as st 
import yaml
import pandas as pd
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler

import statsmodels.api as sm  

import statsforecast
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoTheta, AutoCES
from statsforecast.utils import ConformalIntervals

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import load_model

import datetime

from plotly.subplots import make_subplots
from tabs.my_classes import TransfoMonth, TransfoHour, Multiclass
from catboost import CatBoostClassifier

from pytorch_tabnet.tab_model import TabNetClassifier

# Rajouter le sous-répertoire tabs au path
import os
import sys 
sys.path.append(os.path.normpath(os.path.join(os.getcwd(), "tabs")))

st.set_page_config(layout="wide")

st.title("Accidents routiers en France")

st.sidebar.title("Sommaire")
pages = ["Accueil", "Contexte et objectifs", "Exploration et DataVisualisation", "Modélisation - Séries temporelles", "Classification multiclasse", "Analyse du meilleur modèle", "Optimisations", "Conclusion et perspectives"]
page = st.sidebar.radio("Aller vers", pages)


if page == pages[0]:
    st.write("## Projet fil rouge DataScientest, promotion Septembre 2023")
    from intro_streamlit import onglet_accueil
    onglet_accueil()   
    
if page == pages[1]:
    st.write("### Contexte et objectifs")
    from intro_streamlit import onglet_intro
    onglet_intro()


if page == pages[2]:
    st.write("### Exploration et DataVisualisation")
    from explo_streamlit import explo
    explo()

if page == pages[3]:
    st.write("### Modélisation - Séries temporelles")
    from series_temporelles_streamlit import onglet_series_temporelles
    onglet_series_temporelles()
    
    
if page == pages[4]:
    st.write("### Classification multiclasse")
    from multiclasses_streamlit import onglet_multiclasses
    onglet_multiclasses()


if page == pages[5]:
    st.write("### Analyse du meilleur modèle")
    from analyse_meilleur_modele_streamlit import onglet_analyse_meilleur_modele
    onglet_analyse_meilleur_modele()
    
    
if page == pages[6]:
    st.write("### Optimisation")
    from optimisation_streamlit import onglet_optimisation
    onglet_optimisation()


if page == pages[7]:
    st.write("### Conclusion et perspectives")
    from conclusion_streamlit import onglet_conclusion
    onglet_conclusion()
