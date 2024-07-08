import streamlit as st 
import yaml
import pandas as pd
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm  

import statsforecast
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoTheta, AutoCES
from statsforecast.utils import ConformalIntervals

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import load_model

import datetime

def onglet_analyse_meilleur_modele():

    tab1, tab2 = st.tabs(["Features Importances", "SHAP"])
    
    with tab1 :
        col1, col2 = st.columns([0.5, 0.5])

        with col1 :
            st.image("../data/img/feature_importances.png")
        
        with col2:
            st.write("#")
            st.write("#")
            st.write("**:blue[Interprétation :]**")
            st.write("Random Forest accorde :")
            st.write("• Une très forte importance à **l'utilisation ou non de la ceinture de sécurité**.")
            st.write("• Une forte importance à **l'âge de l'usager**, au **type de collision**, et au fait de **rouler ou non en agglomération**.")
            st.write("• Une importance légèrement moindre à **la latitude**, à **la place dans le véhicule ou piéton** et à **l'obstacle mobile heurté**.")
        

    
    with tab2 :
        modalites = ["Indemnes", "Blessés légers", "Blessés hospitalisés", "Tués"]
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col2 :
            modalite = st.selectbox("**Choisissez la gravité des accidentés :**", modalites)
        
        col1, col2, col3 = st.columns([0.35,  0.35, 0.3])    
            
        if modalite == "Indemnes" :
            with col1 :
                st.image("../data/img/shap1_indemnes.png", width = 350)
            
            with col2 :
                st.image("../data/img/shap2_indemnes.png", width = 350)
        
        if modalite == "Blessés légers" :
            with col1 :
                st.image("../data/img/shap1_bl_legers.png", width = 350)
            
            with col2 :
                st.image("../data/img/shap2_bl_legers.png", width = 350)
    
        if modalite == "Blessés hospitalisés" :
            with col1 :
                st.image("../data/img/shap1_bl_hosp.png", width = 350)
            
            with col2 :
                st.image("../data/img/shap2_bl_hosp.png", width = 350)
        
        if modalite == "Tués" :
            with col1 :
                st.image("../data/img/shap1_tues.png", width = 350)
            
            with col2 :
                st.image("../data/img/shap2_tues.png", width = 350)
     

        with col3 :
            st.write("**:blue[Interprétation :]**")       
            if modalite == "Indemnes" :
                st.write("La probabilité d'être classé indemne augmente avec :")
                st.write("• **Utiliser la ceinture de sécurité**")    
                st.write("• **Rouler en agglomération**") 
                st.write("• **Être conducteur**")
                st.write("• **Ne pas porter de casque**")
                st.write("• **Ne pas heurter d'obstacle fixe**")
                st.write("• **Ne pas être à proximité du point de choc**")
                st.write("• **Être un homme**")
                st.write("• **Circuler sur une route à sens unique**")
                st.write("• **Être jeune**")
                st.write("• **Être immobile ou changer de direction**")
                st.write("• **Utiliser un équipement indéterminé**")
                st.write("• **Être au niveau d'une intersection**")
                st.write("• **Ne pas utiliser de gant**")
                st.write("• **Circuler en 2ième partie de journée**")
            
            if modalite == "Blessés légers" :   
                st.write("La probabilité d'être classé blessé léger augmente avec :")
                st.write("• **Rouler en agglomération**") 
                st.write("• **Être passager ou piéton**")
                st.write("• **Rouler sur une route unidirectionnelle**")
                st.write("• **Être une femme**")   
                st.write("• **Porter un casque**")
                st.write("• **Rouler dans le même sens ou à contre-sens**")
                st.write("• **Utiliser un équipement indétermniné**")
                st.write("• **Être au niveau d'une intersection**")
                st.write("• **Porter des gants**")
                st.write("• **Être sur une route rectiligne**")
            
            if modalite == "Blessés hospitalisés" :       
                st.write("La probabilité d'être classé blessé hospitalisé augmente avec :")
                st.write("• **Ne pas utiliser la ceinture de sécurité**")
                st.write("• **Porter un casque**")
                st.write("• **Rouler hors agglomération**")  
                st.write("• **Circuler sur une route bidirectionnelle**")
                st.write("• **Être passager ou piéton**")
                st.write("• **Heurter un obstacle fixe**")
                st.write("• **Être à proximité du point de choc**")
                st.write("• **Être une femme**")
                st.write("• **Ne pas utiliser d'équipement indéterminé**")
                st.write("• **Utiliser des gants**")
                st.write("• **Circuler sur une route en courbe**")
                  
            
            if modalite == "Tués" :            
                st.write("La probabilité d'être classé tué augmente avec :")
                st.write("• **Rouler hors agglomération**")
                st.write("• **Ne pas utiliser de ceinture de sécurité**")
                st.write("• **Heuter un obstacle fixe**")  
                st.write("• **Être âgé**")
                st.write("• **Ne pas heuter d'obstacle mobile**")
                st.write("• **Circuler sur une route bidirectionnelle**")
                st.write("• **Être à proximité du point de choc**") 
                st.write("• **Ne pas être au niveau d'une intersection**")
                st.write("• **Être un homme**")
                st.write("• **Ne pas utiliser d'équipement indéterminé**")
                st.write("• **Circuler en 1ère partie de journée**")
                st.write("• **Rouler de nuit**")
                st.write("• **Être immobile ou changer de direction**")
                st.write("• **Circuler les premiers mois de l'année**")
                st.write("• **Circuler sur une route en courbe**")  
    
