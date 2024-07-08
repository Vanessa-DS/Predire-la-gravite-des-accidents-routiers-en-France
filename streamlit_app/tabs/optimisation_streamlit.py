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
import zipfile
import tempfile
import os

def onglet_optimisation():

    tab1, tab2, tab3 = st.tabs(["Modélisation avec 2 classes", "Ajout de la variable 'nb_usagers_gr'", "Ajout de la variable 'catv_percute'"])
    
    with tab1 :
        st.write("##### **:blue[Classification binaire : un niveau de gravité contre tous les autres]**")
        modalites = ["Indemnes vs autres", "Blessés légers vs autres", "Blessés hospitalisés vs autres", "Tués vs autres"]
        modalite = st.selectbox("Quelle **classification binaire** de l'état de l'usager souhaitez-vous analyser ?", modalites)
        
        rf_indemne_zipfile = zipfile.ZipFile('../data/saved_models/RandomForest_2classes_indemnes_autres.zip')
        with tempfile.TemporaryDirectory() as tmp_dir:
            rf_indemne_zipfile.extractall(tmp_dir)
            root_folder = rf_indemne_zipfile.namelist()[0]
            rf_indemne_dir = os.path.join(tmp_dir, root_folder)
            rf_indemne_model = joblib.load(rf_indemne_dir)
        rf_bl_legers_zipfile = zipfile.ZipFile('../data/saved_models/RandomForest_2classes_blesseslegers_autres.zip')
        with tempfile.TemporaryDirectory() as tmp_dir:
            rf_bl_legers_zipfile.extractall(tmp_dir)
            root_folder = rf_bl_legers_zipfile.namelist()[0]
            rf_bl_legers_dir = os.path.join(tmp_dir, root_folder)
            rf_bl_legers_model = joblib.load(rf_bl_legers_dir)
        rf_bl_hosp_model = joblib.load('../data/saved_models/RandomForest_2classes_blesseshospitalises_autres.joblib')
        rf_tues_model = joblib.load('../data/saved_models/RandomForest_2classes_tues_autres.joblib')
        
        if modalite == "Indemnes vs autres":
            st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
            rf_indemnes_params = rf_indemne_model.get_params(deep=False)
            rf_indemnes_params = pd.DataFrame.from_dict(rf_indemnes_params, orient='index').astype(str)
            rf_indemnes_params = rf_indemnes_params.rename(columns={0:'Paramètre'})
            st.write(rf_indemnes_params.T)  
            
            with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/rf_binaire_indemnesvsautres_confusion_matrix.png", width=500)
                

                col1.write("Le score est de 0.8541 sur l'ensemble d'entraînement, et de 0.8012 sur l'ensemble de test.")
                rf_indemne_cr = pd.read_csv("../data/img/rf_binaire_indemnesvsautres_report.csv")
                col1.dataframe(rf_indemne_cr)
            
            with st.expander("**:green[INTERPRETABILITE AVEC SHAP]**" , expanded=False):
                col1, col2, col3= st.columns([0.33,0.33,0.33])   
                col1.image("../data/img/shap1_indemnes_binaire.png", width=300)  
                col2.image("../data/img/shap2_indemnes_binaire.png", width=300)  
                col3.write("Les paramètres les plus influents sont :")
                col3.markdown("• l'utilisation d' une ceinture de sécurité")
                col3.markdown("• être à l'avant du véhicule")
                col3.markdown("• ne pas porter de casque (certainement lié au fait de ne pas être en 2-roues)")
                col3.markdown("• ne pas heurter d'obstacle fixe")

        if modalite == "Blessés légers vs autres":
            st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
            rf_bl_legers_params = rf_bl_legers_model.get_params(deep=False)
            rf_bl_legers_params = pd.DataFrame.from_dict(rf_bl_legers_params, orient='index').astype(str)
            rf_bl_legers_params = rf_bl_legers_params.rename(columns={0:'Paramètre'})
            st.write(rf_bl_legers_params.T)  
            
            with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/rf_binaire_bllegersvsautres_confusion_matrix.png", width=500)
                

                col1.write("Le score est de 0.8054 sur l'ensemble d'entraînement, et de 0.7278 sur l'ensemble de test.")
                rf_bl_legers_cr = pd.read_csv("../data/img/rf_binaire_bllegersvsautres_report.csv")
                col1.dataframe(rf_bl_legers_cr)
            
            with st.expander("**:green[INTERPRETABILITE AVEC SHAP]**" , expanded=False):
                col1, col2, col3= st.columns([0.33,0.33,0.33])   
                col1.image("../data/img/shap1_bl_legers_binaire.png", width=300)  
                col2.image("../data/img/shap2_bl_legers_binaire.png", width=300)  
                col3.write("Les paramètres les plus influents sont :")
                col3.markdown("• être à l'arrière du véhicule") 
                col3.markdown("• être une femme")
                col3.markdown("• ne pas utiliser de ceinture de sécurité")    
                col3.markdown("• porter un casque (certainement lié au fait d'être en 2-roues)")
        
        if modalite == "Blessés hospitalisés vs autres":
            st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
            rf_bl_hosp_params = rf_bl_hosp_model.get_params(deep=False)
            rf_bl_hosp_params = pd.DataFrame.from_dict(rf_bl_hosp_params, orient='index').astype(str)
            rf_bl_hosp_params = rf_bl_hosp_params.rename(columns={0:'Paramètre'})
            st.write(rf_bl_hosp_params.T)  
            
            with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/rf_binaire_blhospvsautres_confusion_matrix.png", width=500)
                

                col1.write("Le score est de 0.8560 sur l'ensemble d'entraînement, et de 0.8296 sur l'ensemble de test.")
                rf_bl_legers_cr = pd.read_csv("../data/img/rf_binaire_blhospvsautres_report.csv")
                col1.dataframe(rf_bl_legers_cr)
            
            with st.expander("**:green[INTERPRETABILITE AVEC SHAP]**" , expanded=False):
                col1, col2, col3= st.columns([0.33,0.33,0.33])   
                col1.image("../data/img/shap1_bl_hosp_binaire.png", width=300)  
                col2.image("../data/img/shap2_bl_hosp_binaire.png", width=300)  
                col3.write("Les paramètres les plus influents sont :")
                col3.markdown("• ne pas utiliser de ceinture de sécurité")
                col3.markdown("• rouler hors agglomération") 
                col3.markdown("• circuler sur une route bidirectionnelle")            
        
        if modalite == "Tués vs autres":
            st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
            rf_tues_params = rf_tues_model.get_params(deep=False)
            rf_tues_params = pd.DataFrame.from_dict(rf_tues_params, orient='index').astype(str)
            rf_tues_params = rf_tues_params.rename(columns={0:'Paramètre'})
            st.write(rf_tues_params.T)  
            
            with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/rf_binaire_tuesvsautres_confusion_matrix.png", width=500)
                

                col1.write("Le score est de 0.9596 sur l'ensemble d'entraînement, et de 0.9406 sur l'ensemble de test.")
                rf_tues_cr = pd.read_csv("../data/img/rf_binaire_tuesvsautres_report.csv")
                col1.dataframe(rf_tues_cr)
            
            with st.expander("**:green[INTERPRETABILITE AVEC SHAP]**" , expanded=False):
                col1, col2, col3= st.columns([0.33,0.33,0.33])   
                col1.image("../data/img/shap1_tues_binaire.png", width=300)  
                col2.image("../data/img/shap2_tues_binaire.png", width=300)  
                col3.write("Les paramètres les plus influents sont :")
                col3.markdown("• ne pas utiliser de ceinture de sécurité") 
                col3.markdown("• rouler hors agglomération") 
                col3.markdown("• circuler sur une route bidirectionnelle")    
        
    with tab2 :
        st.write("##### **:blue[Création de la variable 'nb_usagers_gr']**")
        st.write("Chaque accident est enregistré dans la variable 'Num_Acc' qui est identique pour toutes les personnes concernées par cet accident.")
        st.write("En regroupant, le nombre de personnes concernées par un même accident, on crée la variable **'nb_usagers'**.")
        
        col1, col2 = st.columns([0.5, 0.5])
        with col1 :
            st.image('../data/img/nb_usagers.png')
            
        with col2 :
            st.image('../data/img/nb_usagers2.png')
            
        st.write("90% du jeu de données a un nombre d'usagers accidentés par accident inférieur ou égal à 5.")
        st.write("Regroupement du nombre d'usagers pour obtenir la variable **'nb_usagers_gr'** avec les modalités **1**, **2**, **3**, **4** ou **5 et +** usagers")
        
        st.write("#")
        st.write("#")
        st.write("##### **:blue[Modélisation avec le meilleur modèle]**")
        st.write("**Amélioration de 0.45% de l'accuracy**")
        
        with st.expander("Performances du modèle et Matrice de confusion" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/rf_4classes_nb_usagers_gr_confusion_matrix.png")
            

            col1.write("Le score est de 0.6594 sur l'ensemble d'entraînement, et de 0.6158 sur l'ensemble de test.")
            rf_cr = pd.read_csv("../data/img/rf_4classes_nb_usagers_gr_report.csv")
            col1.dataframe(rf_cr)
        
    with tab3 :
        st.write("##### **:blue[Création de la variable 'catv_percute']**")
        
        st.write("La variable 'catv' correspond à 2 choses différentes selon si la personne accidentée est :")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• dans un véhicule : 'catv' est la catégorie de véhicule dans lequel est la personne")
            st.write("• piéton : 'catv' est la catégorie de véhicule qui a percuté le piéton")
        
        st.write("#")
        st.write("Les variables 'obs' et 'obsm' correspondent à 2 choses différentes selon si la personne accidentée est :")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• dans un véhicule : 'obs' et 'obsm' sont les obstacles fixes et mobiles percutés par le véhicule dans lequel est la personne")
            st.write("• piéton : 'obs' et 'obsm' sont les obstacles fixes et mobiles percutés est le véhicule qui a percuté le piéton")
            
        st.write("#")
        st.write("Création de 'catv_percute' et modification de 'catv', 'obs' et 'obsm'")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("1) Création de la variable 'catv_percute' qui indique le véhicule qui a percuté la personne accidenté pour les piétons")
            st.write("2) Ajout de la modalité 6 ('inconnu') dans 'catv_percute' pour les personnes accidentées autres que les piétons")
            st.write("3) Ajout de la modlaité 6 ('piéton') dans 'catv' pour les piétons")
            st.write("4) Mise à jour des valeurs de 'obs' et 'obsm' à 0 (pas d'obstacle heurté) pour les piétons")
            
        
        st.write("#")
        st.write("#")
        st.write("##### **:blue[Modélisation avec le meilleur modèle]**")
        st.write("**Amélioration de 0.14% de l'accuracy** mais les modifications ne concernent que les piétons soient 7,62% du jeu de données")
        
        with st.expander("Performances du modèle et Matrice de confusion" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/rf_4classes_catv_percute_confusion_matrix.png")
            

            col1.write("Le score est de 0.6583 sur l'ensemble d'entraînement, et de 0.6136 sur l'ensemble de test.")
            rf_cr = pd.read_csv("../data/img/rf_4classes_catv_percute_report.csv")
            col1.dataframe(rf_cr)


