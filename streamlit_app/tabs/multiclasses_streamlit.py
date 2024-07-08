import streamlit as st 
import yaml
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import zipfile
import tempfile
import tensorflow as tf


from my_classes import TransfoMonth, TransfoHour, Multiclass
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from catboost import CatBoostClassifier

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from pickle import load

import xgboost as xgb

@st.cache_data
def load_models():
    ### Chargement des modèles
    catboost_model = CatBoostClassifier()  #définition sans paramètre avant recharge
    catboost_model.load_model('../data/saved_models/catboost_model')
    logreg_model = joblib.load('../data/saved_models/logreg_model')
    svm_model = joblib.load('../data/saved_models/svm_model')
    rf_model = joblib.load('../data/saved_models/RandomForest_4classes.joblib')
    knn_model = joblib.load('../data/saved_models/KNN_best.joblib')
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('../data/saved_models/xgboost_streamlit.json')
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model('../data/saved_models/tabnet_model.zip')
    keras_zipfile = zipfile.ZipFile('../data/saved_models/dnn_keras.zip')
    with tempfile.TemporaryDirectory() as tmp_dir:
        keras_zipfile.extractall(tmp_dir)
        root_folder = keras_zipfile.namelist()[0]
        dnn_model_dir = os.path.join(tmp_dir, root_folder)
        dnn_keras_model = tf.keras.models.load_model(dnn_model_dir)
    dnn_pytorch_model = Multiclass(dropout_rate=0.1, n_neurones=800)
    dnn_pytorch_model.load_state_dict(torch.load("../data/saved_models/dict_model_pytorch_opt.pt"))
    dnn_pytorch_model.eval()
    
    preprocessor_keras = load(open('../data/saved_models/' + 'preprocessor_keras.pkl','rb'))
    preprocessor_tabnet = load(open('../data/saved_models/' + 'preprocessor_tabnet.pkl','rb'))
    preprocessor_pytorch = load(open('../data/saved_models/' + 'preprocessor_pytorch.pkl','rb'))
    preprocessor_knn = load(open('../data/saved_models/' + 'preprocessor_knn.pkl','rb'))
    preprocessor_xgb = load(open('../data/saved_models/' + 'preprocessor_xgb.pkl','rb'))
    
    return catboost_model, logreg_model, svm_model, rf_model, knn_model, xgb_model, tabnet_model, dnn_keras_model, dnn_pytorch_model,preprocessor_keras, preprocessor_tabnet,preprocessor_pytorch,preprocessor_knn,preprocessor_xgb

def onglet_multiclasses():

    catboost_model, logreg_model, svm_model, rf_model, knn_model,  xgb_model, tabnet_model, dnn_keras_model,dnn_pytorch_model,preprocessor_keras, preprocessor_tabnet,preprocessor_pytorch,preprocessor_knn,preprocessor_xgb = load_models()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Régression logistique", "LinearSVC", "KNN", "Random Forest", "CatBoost", "XGBoost", "DNN-MLP","DNN with Transformers","Comparaison des modèles", "Faire une prédiction"])

    with tab1:
        st.header("Régression logistique")
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
           st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
           st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
           st.markdown("• Normalisation de l'âge par MinMaxScaler")
           st.markdown("• Encodage des variables catégorielles")
        
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        log_reg_params = logreg_model.named_steps['log_reg'].get_params(deep=False)
        log_reg_params = pd.DataFrame.from_dict(log_reg_params, orient='index')
        log_reg_params = log_reg_params.rename(columns={0:'Paramètre'})
        st.write(log_reg_params.T)
        
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/logreg_confusion_matrix.jpg", width=600)
            

            col1.write("Le score est de 0.6206 sur l'ensemble d'entraînement, et de 0.6225 sur l'ensemble de test.")
            logreg_cr = pd.read_csv("../data/img/logreg_report.csv")
            col1.dataframe(logreg_cr)
        
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            st.write("L'analyse des coefficients du modèle (validation croisée sur 3 sous-ensembles) montre une forte influence :")
            st.markdown("• du fait d'être un piéton (place_rec = 4.0)")
            st.markdown("• d'être dans une configuration où un piéton a été heurté par un véhicule (obsm = 1.0)")
            st.markdown("• des catégories de véhicules impliqués, avec dans l'ordre, les vélos, motos, poids lourds, autres, transports en commun et voitures")
            st.markdown("• la présence d'une piste cyclable (situ_5.0)")
            st.image("../data/img/logreg_coefficients.png", width=500)
            st.write("On note également la faible influence des variables continues.")
 
        

    with tab2:
        st.header("LinearSVC, après une approximation de type Nystroëm pour introduire des non-linéarités dans le modèle")
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
           st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
           st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
           st.markdown("• Normalisation de l'âge par MinMaxScaler")
           st.markdown("• Encodage des variables catégorielles")
           
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        nys_params = svm_model.named_steps['nystroem'].get_params(deep=False)
        nys_params = pd.DataFrame.from_dict(nys_params, orient='index')
        nys_params = nys_params.rename(columns={0:"Paramètres de l'approximation Nystroëm"})
        st.write(nys_params.T)
        svm_params = svm_model.named_steps['svm'].get_params(deep=False)
        svm_params = pd.DataFrame.from_dict(svm_params, orient='index')
        svm_params = svm_params.rename(columns={0:'Paramètres du modèle LinearSVC'})
        st.write(svm_params.T)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/svm_confusion_matrix.jpg", width=600)
            
            col1.write("Le score est de 0.6248 sur l'ensemble d'entraînement, et de 0.6242 sur l'ensemble de test.")
            svm_cr = pd.read_csv("../data/img/svm_report.csv")
            col1.dataframe(svm_cr)
            
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            st.write("""Ce modèle est difficilement interprétable. Il a pour principal intérêt de montrer 
                         que la prise en compte de non-linéarités améliore les performances du modèle.""")
            
        
            

    with tab3:
        st.header("KNN")
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
           st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
           st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
           st.markdown("• Standardisation de l'âge par Standard Scaler")
           st.markdown("• Encodage des variables catégorielles")
           
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        knn_params = knn_model.get_params(deep=False)
        knn_params = pd.DataFrame.from_dict(knn_params, orient='index')
        knn_params = knn_params.rename(columns={0:'Paramètres du modèle KNN'})
        st.write(knn_params.T)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/knn_confusion_matrix.jpg", width=500)
            
            col1.write("Le score est de 0.6161 sur l'ensemble d'entraînement, et de 0.5655 sur l'ensemble de test.")
            knn_cr = pd.read_csv("../data/img/knn_report.csv")
            col1.dataframe(knn_cr)
            
 
        
    with tab4:
        st.header("Random Forest Classifier")
        with st.expander("**:green[PREPROCESSING ET SELECTION DES VARIABLES]**" , expanded=True):
            st.write(" Pas de preprocessing particulier (ni normalisation/standardisation des variables continues, ni encodage des variables catégorielles)")
            st.write("""Ce modèle n'utilise pas la variable indiquant si un siège a été utilisé comme équipement de sécurité. Toutes
                     les autres variables sont retenues.""")
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        rf_params = rf_model.get_params(deep=False)
        rf_params = pd.DataFrame.from_dict(rf_params, orient='index')
        rf_params = rf_params.rename(columns={0:'Paramètre'})
        st.write(rf_params.T)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/rf_confusion_matrix.png", width=600)
            

            col1.write("Le score est de 0.6560 sur l'ensemble d'entraînement, et de 0.6125 sur l'ensemble de test.")
            rf_cr = pd.read_csv("../data/img/rf_report.csv")
            col1.dataframe(rf_cr)
            
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            st.write("Ce modèle ayant obtenu les meilleures performances, la section 'Analyse du meilleur modèle' lui est consacrée.")
        
    with tab5:
        st.header("CatBoost Classifier")
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
            st.markdown("Pas de preprocessing particulier (ni normalisation/standardisation des variables continues, ni encodage des variables catégorielles)")
           
           
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        catb_params = catboost_model.get_params(deep=False)
        catb_params = pd.DataFrame.from_dict(catb_params, orient='index')
        catb_params = catb_params.rename(columns={0:'Paramètre'})
        st.write(catb_params.T)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/catboost_confusion_matrix.jpg", width=600)
            

            col1.write("Le score est de 0.6398 sur l'ensemble d'entraînement, et de 0.6171 sur l'ensemble de test.")
            catboost_cr = pd.read_csv("../data/img/catboost_report.csv")
            col1.dataframe(catboost_cr)
            
        
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            methode_interp = st.selectbox("Par quelle méthode souhaitez-vous quantifier l'importance des variables explicatives ? :", ["Feature importance", "SHAP"]) 
            if methode_interp == "Feature importance" :
                col1, col2= st.columns([0.65,0.35])   
                col2.write("Les paramètres les plus influents sont :")
                col2.markdown("• le fait d'utiliser une ceinture de sécurité")
                col2.markdown("• le type de collision (notamment, si plusieurs véhicules sont impliqués)")
                col2.markdown("• les catégories des véhicules impliqués")
                col2.markdown("• la localisation de l'accident (latitude, puis longitude dans une moindre mesure)")
                col2.markdown("• la place de l'usager")
                col2.markdown("• la présence d'un obstacle mobile")
                col2.markdown("• la catégorie de route")
                col2.markdown("• l'âge de l'usager")
                col2.markdown("• le fait d'être en agglomération")
                col1.image("../data/img/catboost_feature_importance.jpg", width=600)   
                
            if methode_interp == "SHAP" :
                modalites = ["Indemne", "Blessé léger", "Blessé hospitalisé", "Tué"]
                modalite = st.selectbox("Quel **niveau de gravité** de l'état de l'usager souhaitez-vous analyser ?", modalites)
                
                if modalite == "Indemne":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/catboost_shap_indemnes.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **indemne** sont :")
                    col2.write("• s'il n'est pas dans un véhicule de catégorie vélo, trottinette, transport en commun ou autres")
                    col2.write("• s'il utilise la ceinture de sécurité")
                    col2.write("• s'il occupe la place de conducteur")
                    col2.write("• s'il roule en agglomération")
                    col2.write("• s'il est un homme ")
                    col2.write("• s'il n'est pas à proximité du point de choc")
                    col2.write("• s'il est jeune")
                if modalite == "Blessé léger":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/catboost_shap_blesses.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **blessé léger** sont :")
                    col2.write("• le fait d'être une femme")
                    col2.write("• le fait de ne pas être à l'avant du véhicule")
                    col2.write("• la présence d'un obstacle mobile")
                    col2.write("• de ne pas être âgé")
                    col2.write("• de rouler en agglomération")
                    col2.write("• d'être dans un régime de circulation unidirectionnel ")
                    col2.write("• d'être sur une route avec un état de surface non détérioré")
                    col2.write("• de ne pas rouler de nuit")
                if modalite == "Blessé hospitalisé":
                    col1, col2 = st.columns([0.5,0.5])   
                    col1.image("../data/img/catboost_shap_blessesgraves.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **blessé hospitalisé** sont :")
                    col2.write("• la non utilisation de la ceinture de sécurité")
                    col2.write("• le fait de rouler hors agglomération")
                    col2.write("• le fait de ne pas être à l'avant du véhicule")
                    col2.write("• la présence à proximité du point de choc")
                    col2.write("• la présence d'un obstacle fixe")
                    col2.write("• le fait de ne pas être jeune")
                    col2.write("• la présence d'un régime de circulation bidirectionnelle")
                    col2.write("• le fait d'être une femme")
                    col2.write("• le fait de circuler en début de journée")
                    col2.write("• le fait de porter un casque (certainement car cycliste, donc usager de catégorie catv=4, donc plutôt forte)")
                if modalite == "Tué":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/catboost_shap_tues.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **tué** sont :")
                    col2.write("• la non utilisation de la ceinture de sécurité")
                    col2.write("• le fait de rouler hors agglomération")
                    col2.write("• le fait d'être âgé")
                    col2.write("• la circulation sur une route de type autoroute, route nationale, route départementale")
                    col2.write("• la présence d'un obstacle fixe")
                    col2.write("• le fait d'être à proximité du point de choc")
                    col2.write("• la présence d'un régime de circulation bidirectionnelle")
                    col2.write("• le fait de ne pas être à l'avant du véhicule")
                    col2.write("• le fait d'être un homme")
                    col2.write("• l'absence d'intersection")
                    col2.write("• la présence d'un régime de circulation bidirectionnelle")
                    
    with tab6:
        st.header("XGBoost Classifier")  
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
            st.write("Même si le modèle ne souffre pas de l'obligation d'avoir des données normalisées, par cohérence avec d'autres modèles, le preprocessing suivant a été mis en oeuvre :")
            st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
            st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
            st.markdown("• Normalisation de l'âge par MinMaxScaler")

           
        xgboost_cr = pd.read_csv("../data/img/xgboost_report.csv")  
        st.write("Après optimisation, le jeu de paramètres ayant conduit aux meilleures performances est le suivant :")
        
        st.write(xgb_model)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/xgboost_confusion_matrix.jpg", width=600)
            

            col1.write("Le score est de 0.5764 sur l'ensemble d'entraînement, et de 0.5724 sur l'ensemble de test.")
            xgb_cr = pd.read_csv("../data/img/xgboost_report.csv")
            col1.dataframe(xgb_cr)
            
        
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            xgb_interp = st.selectbox("Par quelle méthode souhaitez-vous quantifier l'importance des variables explicatives ? :", ["Features importance", "SHAP"]) 
            if xgb_interp == "Features importance" :
                col1, col2, col3 = st.columns(3)   
                col1.image("../data/img/xgb_importance_weight.png", width=250)
                col2.image("../data/img/xgb_importance_gain.png", width=300)
                col3.image("../data/img/xgb_importance_cover.png", width=300)
                
            if xgb_interp == "SHAP" :
                modalites_xgb = ["Indemnes", "Blessés légers", "Blessés hospitalisés", "Tués"]
                modalite = st.selectbox("Quel **niveau de gravité** de l'état de l'usager souhaitez-vous analyser ?", modalites_xgb)
                
                if modalite == "Indemnes":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/xgboost_shap_indemnes.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **indemne** sont :")
                    col2.write("• l'utilisation de la ceinture de sécurité")
                    col2.write("• l'absence d'obstacle fixe")
                    col2.write("• le fait de rouler en agglomération")
                    col2.write("• de ne pas être à proximité du point de choc")
                    col2.write("• d'être plutôt jeune")
                if modalite == "Blessés légers":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/xgboost_shap_bllegers.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **blessé léger** sont :")
                    col2.write("• le fait d'être une femme")
                    col2.write("• de ne pas être âgé")
                    col2.write("• de porter une ceinture de sécurité")
                    col2.write("• de rouler en agglomération")
                    col2.write("• la présence d'un obstacle mobile")
                    col2.write("• le fait de ne pas être à l'avant")
                if modalite == "Blessés hospitalisés":
                    col1, col2 = st.columns([0.5,0.5])   
                    col1.image("../data/img/xgboost_shap_blgraves.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **blessé hospitalisé** sont :")
                    col2.write("• le fait de rouler hors agglomération")
                    col2.write("• la non utilisation de la ceinture de sécurité")
                    col2.write("• le fait d'être un homme")
                    col2.write("• la présence d'un régime de circulation bidirectionnelle")
                    col2.write("• le fait de ne pas être à l'avant du véhicule")
                    col2.write("• la présence à proximité du point de choc")
                    col2.write("• la présence d'un obstacle fixe")
                    col2.write("• le fait de porter un casque (certainement car cycliste, donc usager de catégorie catv=4, donc plutôt forte)")
                if modalite == "Tués":
                    col1, col2= st.columns([0.5,0.5])   
                    col1.image("../data/img/xgboost_shap_tues.png", width =400)  
                    col2.write("Les variables pouvant expliquer que l'usager est **tué** sont :")
                    col2.write("• le fait de rouler hors agglomération")
                    col2.write("• la non utilisation de la ceinture de sécurité")
                    col2.write("• le fait d'être âgé")
                    col2.write("• la présence d'un obstacle fixe")
                    col2.write("• une circulation nocturne")
                    col2.write("• l'absence d'intersection")
                    col2.write("• la proximité du point de choc")
                    col2.write("• la présence d'un régime de circulation bidirectionnelle")
                    col2.write("• le fait de ne pas être à l'avant du véhicule")
                    col2.write("• le fait d'être un homme")
        
    with tab7:
        st.header("Réseaux de neurones profonds - Perceptron multicouche (MLP)") 
        dnn_framework = st.selectbox("Vous êtes plutôt Keras ? ou Pytorch ? :", ["Keras", "Pytorch"])
        
        dnn_keras_cr = pd.read_csv("../data/img/DNN_4classes_renumok_report.csv")
        dnn_pytorch_cr = pd.read_csv("../data/img/dnn_pytorch_report.csv")
        
        if dnn_framework == 'Keras':  
            with st.expander("**:green[PREPROCESSING]**", expanded=True):
                st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
                st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
                st.markdown("• Standardisation de l'âge par Standard Scaler")
                
            st.write("Après optimisation, le modèle ayant conduit aux meilleures performances est le suivant :")
            col1, col2 = st.columns([0.6, 0.4])
            with col1 :
                st.write(dnn_keras_model.summary(print_fn=lambda x: st.text(x)))
                #st.image("../data/img/DNN_Keras_summary.png", width=600)
                
            with col2 :
                st.write("#")         
                st.write("#") 
                st.write(":blue[**Modèle avec :**]")
                col1, col2 = st.columns([0.02, 0.98])
                with col2 :
                    st.write("• activation : **gelu** pour les couches dense_0 à dense_4 et **Softmax** pour la couche dense_5")
                    st.write("• kernel_initializer : **tf.keras.initializers.GlorotNormal()** pour les couches dense_0 à dense_4")
            
            st.write("#")
            col1, col2 = st.columns([0.5, 0.5])   
            with col1 :     
                st.write(":blue[**Compilation du modèle avec :**]")
            
            with col2 :          
                st.write(":blue[**Entraînement du modèle avec :**]")
                
            col1, col2, col3, col4 = st.columns([0.02, 0.48, 0.02, 0.48])
            with col2 :
                st.write("• loss = **tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, ignore_class=None, reduction='sum_over_batch_size', name='sparse_categorical_crossentropy')**")
                st.write("• optimizer : **adam**")
                st.write("• metrics : **sparse_categorical_accuracy**")
            
            with col4 :
                st.write("• epochs : **70**")
                st.write("• batch_size : **128**")
                st.write("• validation_split : **0.1**")
                st.write("• callbacks : **ReduceLROnPlateau(monitor = 'val_loss', min_delta = 0.01, patience = 5, factor = 0.5, cooldown = 2, verbose = 1)**")
                st.write("• class_weight : **{Indemnes: 0.6057177536467477, Blessés légers: 0.6196864536443666, Blessés hospitalisés: 1.5969447037086422, Tués: 9.161621680690635,}**")
            
            with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/DNN_4classes_confusion_matrix.png", width=600)
                
                col1.write("Le score est de 0.5961 sur l'ensemble d'entraînement, et de 0.5852 sur l'ensemble de test.")
                dnn_keras_cr = pd.read_csv("../data/img/DNN_4classes_renumok_report.csv")
                col1.dataframe(dnn_keras_cr)
            
        if dnn_framework == 'Pytorch':
            with st.expander("**:green[PREPROCESSING]**", expanded=True):
                st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
                st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
                st.markdown("• Normalisation de l'âge par MinMaxScaler")
                  
            st.write("Après optimisation, le modèle ayant conduit aux meilleures performances est le suivant :")
            col1, col2 = st.columns([0.5, 0.5])
            with col1 :
                st.write(dnn_pytorch_model.eval())
                
            with col2 :          
                st.write(":blue[**Modèle avec :**]")
                col1, col2 = st.columns([0.02, 0.98])
                with col2 :
                    st.write("• activation : **ReLU** pour les couches denses et **Softmax** pour la dernière couche")
                    st.write("• Des couches de Dropout, avec un **taux de de désactivation de 10%** ont été intercalées pour contrer le surapprentissage.")
                    st.write("• L'influence de ce taux a été analysée par validation croisée.")
                    st.image("../data/img/dnn_pytorch_dropout.png", width=300)
            
            st.write("#")
            col1, col2 = st.columns([0.5, 0.5])   
            with col1 :     
                st.write(":blue[**Compilation du modèle avec :**]")
            
            with col2 :          
                st.write(":blue[**Entraînement du modèle avec :**]")
                
            col1, col2, col3, col4 = st.columns([0.02, 0.48, 0.02, 0.48])
            with col2 :
                st.write("• loss = **nn.CrossEntropyLoss(weight=class_weights, reduction='mean')**")
                st.write("• optimizer : **SGD**")
                
            
            with col4 :
                st.write("• epochs : **31**")
                st.write("• batch_size : **64**")
                st.write("• validation_split : **0.1**")
                st.write("• callbacks : **ReduceLROnPlateau(optimizer, mode='max', factor=0.25, patience=2, threshold=0.001, min_lr=1e-4)**")
                st.write("• class_weight : **tensor([0.6057, 0.6197, 1.5969, 9.1614])**")
            
            with st.expander("Performances du modèle et Matrice de confusion" , expanded=True):
                col1, col2= st.columns(2)
                col2.image("../data/img/dnn_pytorch_opt_confusion_matrix.jpg", width=600)
                
                col1.write("Le score est de 0.6060 sur l'ensemble d'entraînement, et de 0.5929 sur l'ensemble de test.")
                dnn_keras_cr = pd.read_csv("../data/img/dnn_pytorch_opt_report.csv")
                col1.dataframe(dnn_keras_cr)
        
    
    with tab8:
        st.header("Réseaux de neurones profonds avec transformers") 
        st.write("**TabNet**, développée par Google, est spécialisée pour les données tabulaires et a pour objectif de concurrencer les algorithmes d'arbres boostés.")
        st.write("A pour intérêt de proposer des méthodes d'interprétation des résultats obtenus.")
        with st.expander("**:green[PREPROCESSING]**", expanded=True):
                st.markdown("• Transformation des heures et des mois : position sur un cercle unité en utilisant 2 entrées, sinus et cosinus.")
                st.markdown("• Normalisation des latitudes et longitudes par Robust Scaler")
                st.markdown("• Normalisation de l'âge par MinMaxScaler")
        col1, col2 = st.columns([0.55,0.45])
        col1.write("L'architecture de TabNet est la suivante [Arik and Pfister, 2021] :")
        col1.write("Sélection des variables par **attention séquentielle**, à chaque batch (et non sur l'ensemble des données.)")
        col1.image("../data/img/TabNet_layers.png", width=500)
        
        
                
        tabnet_params = tabnet_model.get_params(deep=False)
        col2.write("Après optimisation (via Optuna), le modèle ayant conduit aux meilleures performances a les paramètres suivants :")
        col2.write(tabnet_params)
        
        with st.expander("**:green[PERFORMANCES DU MODELE ET MATRICE DE CONFUSION]**" , expanded=True):
            col1, col2= st.columns(2)
            col2.image("../data/img/tabnet_confusion_matrix.jpg", width=600)
            
            col1.write("Le score est de 0.6076 sur l'ensemble d'entraînement, et de 0.5957 sur l'ensemble de test.")
            tabnet_cr = pd.read_csv("../data/img/tabnet_report.csv")
            col1.dataframe(tabnet_cr)
        tabnet_cr = pd.read_csv("../data/img/tabnet_report.csv")
        
        with st.expander("**:green[INTERPRETABILITE]**" , expanded=False):
            col1, col2 = st.columns([0.65,0.35])
            col1.image("../data/img/tabnet_feature_importance.jpg", width=600)
            col2.write("Les variables explicatives principalement influentes dans ce modèle sont :")
            col2.markdown("• la catégorie de routes empruntée")
            col2.markdown("• la présence d'un obstacle mobile")
            col2.markdown("• l'utilisation d'équipements de sécurité (le gilet, l'airbag, et dans une moindre mesure, le casque)")
            col2.markdown("• la place occupée par l'usager")
            col2.markdown("• le profil en long de la route")
            col2.markdown("• les coordonnées spatiales, latitude, puis longitude")
            col2.markdown("• la présence d'un obstacle fixe")
            col2.markdown("• l'âge de l'usager")
        
    with tab9:
        st.header("Comparaison des modèles") 
        st.write(""" Pour chaque état de gravité, est présenté ci-dessous un diagramme radar permettant la comparaison des modèles selon 
                 la précision, le rappel et le f1-score.""")
        names_modele = {'catboost' : 'CatBoost',
                        'logreg' : 'Régression logistique',
                        'knn' : 'KNN',
                        'svm' : "LinearSVC + Nystroem",
                        'rf' : 'Random Forest',
                         'xgboost' : 'XGBoost',
                        'dnn_keras' : 'Deep Learning (DNN-Keras)',
                        'dnn_pytorch' : 'Deep Learning (DNN-Pytorch)',
                        'tabnet' : 'Deep Learning (TabNet)'}
        categories = ['Precision', "Recall", "F1-Score"]
        cols = px.colors.DEFAULT_PLOTLY_COLORS
        
        col1, col2= st.columns(2)
        col1.header('Indemnes')
        fig = go.Figure()
        opacity=0.8
        cpt = 0
        for modele in ['logreg', 'svm', 'knn', 'rf', 'catboost', 'xgboost', 'dnn_keras', 'dnn_pytorch', 'tabnet']:
            dataframe = f"{modele}_cr"
            fig.add_trace(go.Scatterpolar(
                line=dict(color=cols[cpt]),
                r=eval(dataframe).iloc[0,1:4],
                theta=categories,
                fill='toself',
                opacity =opacity,
                #legendgroup='1',
                name=names_modele[modele], showlegend=True))
            cpt += 1

        fig.update_layout(
            height=600, 
            width=1000,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    ))
            #legend_tracegroupgap = 10
            )
        col1.plotly_chart(fig, use_container_width=True)
        
        col2.header('Blessés légers')
        fig = go.Figure()
        opacity=0.8
        cpt = 0
        for modele in ['logreg', 'svm', 'knn','rf', 'catboost', 'xgboost', 'dnn_keras', 'dnn_pytorch', 'tabnet']:
            dataframe = f"{modele}_cr"
            fig.add_trace(go.Scatterpolar(
                line=dict(color=cols[cpt]),
                r=eval(dataframe).iloc[1,1:4],
                theta=categories,
                fill='toself',
                opacity =opacity,
                #legendgroup='1',
                name=names_modele[modele], showlegend=True))
            cpt += 1

        fig.update_layout(
            height=600, 
            width=1000,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    ))
            #legend_tracegroupgap = 10
            )
        col2.plotly_chart(fig, use_container_width=True)
        
        col3, col4= st.columns(2)
        col3.header('Blessés graves')
        fig = go.Figure()
        opacity=0.8
        cpt = 0
        for modele in ['logreg', 'svm', 'knn', 'rf', 'catboost', 'xgboost', 'dnn_keras', 'dnn_pytorch', 'tabnet']:
            dataframe = f"{modele}_cr"
            fig.add_trace(go.Scatterpolar(
                line=dict(color=cols[cpt]),
                r=eval(dataframe).iloc[2,1:4],
                theta=categories,
                fill='toself',
                opacity =opacity,
                #legendgroup='1',
                name=names_modele[modele], showlegend=True))
            cpt += 1

        fig.update_layout(
            height=600, 
            width=1000,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    ))
            #legend_tracegroupgap = 10
            )
        col3.plotly_chart(fig, use_container_width=True)
        
        col4.header('Tués')
        fig = go.Figure()
        opacity=0.8
        cpt = 0
        for modele in ['logreg', 'svm', 'knn', 'rf', 'catboost',  'xgboost','dnn_keras', 'dnn_pytorch', 'tabnet']:
            dataframe = f"{modele}_cr"
            fig.add_trace(go.Scatterpolar(
                line=dict(color=cols[cpt]),
                r=eval(dataframe).iloc[3,1:4],
                theta=categories,
                fill='toself',
                opacity =opacity,
                #legendgroup='1',
                name=names_modele[modele], showlegend=True))
            cpt += 1

        fig.update_layout(
            height=600, 
            width=1000,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    ))
            #legend_tracegroupgap = 10
            )
        col4.plotly_chart(fig, use_container_width=True)
        
        st.write(""" L'influence du déséquilibre des classes sur les performances des modèles est ici flagrante : moins une classe est présente 
                 dans la base d'entraînement, moins les performances sont satisfaisantes. Ainsi, les métriques associées à la classification
                 des tués sont les plus basses. """)
        st.write(""" Il est à noter également un gain relativement faible des performances en fonction des différents modèles utilisés, ce qui
                 peut être lié à la complexité de la base de données, et notamment à l'absence de variables fortement informatives.  """)
        st.write(""" Les modèles de Deep Learning n'ont pas permis d'améliorer les performances comparativement aux meilleurs modèles de Machine Learning. """)
        

    with tab10:

        col1, col2, col3 = st.columns(3)

        sexe = col1.selectbox("Sexe de l'usager",('Homme','Femme'))
        dict_sexe = {'0.0': 'Homme', '1.0': 'Femme'}
        
        place_rec = col1.selectbox("Place occupée par l'usager",('Conducteur', 'Passager avant', 'Passager arrière', 'Piéton'))
        dict_place_rec = {'1.0': 'Conducteur', '2.0': 'Passager avant', '3.0': 'Passager arrière', '4.0': 'Piéton'}
        
        age = col1.slider("Age au moment de l'accident", 0 ,112, 18)
        options = col1.multiselect("Equipements de sécurité utilisés",    
                                ["Ceinture de sécurité", "Casque", "Dispositif enfant", "Gilet réfléchissant", "Airbag", "Gants", "Autres", "Indéterminables"])
        dict_eq ={'eq_ceinture':"Ceinture de sécurité", 'eq_casque':"Casque", 'eq_siege':"Dispositif enfant", 'eq_gilet': "Gilet réfléchissant", 'eq_airbag':"Airbag", 'eq_gants':"Gants", 'eq_indetermine': "Indéterminables", 'eq_autre': "Autres"}
        
        latitude = col1.number_input("Latitude", step=1e-6,format="%.6f")
        longitude = col1.number_input("Longitude", step=1e-6,format="%.6f")
        d = col1.date_input("Date de l'accident", value=None)
        t = col1.time_input("Heure de l'accident")
        jour_chome = col1.checkbox("L'accident a eu lieu un jour férié, ou de vacances scolaires. (cochez si oui)")
        week_end = col1.checkbox("L'accident a eu lieu un jour de week end. (cochez si oui)")
        #latitude = col1.number_input("Latitude", min_value=18, max_value=100, step=1)
        
        catv = col2.selectbox("Catégorie du véhicule (impliqué dans l'accident si piéton)",
                            ('Voitures','Motos', "Poids lourds", "Transports en commun", "Vélos / Trottinettes", "Autres"))
        dict_catv = {'0.0':'Voitures','1.0':'Motos', '2.0':"Poids lourds", '3.0':"Transports en commun", '4.0':"Vélos / Trottinettes", '5.0':"Autres"}
        
        motor = col2.selectbox("Type de motorisation du véhicule ",
                            ('Inconnue','Hydrocarbures', "Hybride électrique", "Electrique", "Hydrogène", "Humaine", "Autre"))
        dict_motor = {'0.0':'Inconnue','1.0':'Hydrocarbures', '2.0':"Hybride électrique",'3.0': "Electrique", '4.0':"Hydrogène", '5.0':"Humaine",'6.0': "Autre"}
        
        col = col2.selectbox("Type de collision",("Choc frontal entre 2 véhicules", "Choc par l'arrière - 2 véhicules", "Choc par le côté - 2 véhicules",
                                                "3 véhicules et plus - en chaîne", "3 véhicules et plus - collisions multiples", "Autre collision", "Sans collision"))
        dict_col = {'1.0': "Choc frontal entre 2 véhicules", '2.0': "Choc par l'arrière - 2 véhicules", '3.0': "Choc par le côté - 2 véhicules",
                                                '4.0': "3 véhicules et plus - en chaîne", '5.0': "3 véhicules et plus - collisions multiples", '6.0': "Autre collision", '7.0': "Sans collision"}
        
        prox_pt_choc = col2.checkbox("L'usager concerné se trouvait à proximité du point de choc. (cochez si oui)")
        
        obs = col2.checkbox("Un obstacle fixe a été heurté (cochez si oui).")
        
        obsm = col2.selectbox("Obstacle mobile heurté ? ",('Aucun','Piéton', "Animaux / Autres", "Véhicule"))
        dict_obsm = {'0.0':'Aucun','1.0':'Piéton', '3.0':"Animaux / Autres", '2.0':"Véhicule"}    
        
        manv = col2.selectbox("Manoeuvre principale avant l'accident",
                            ('Même sens de circulation','Circulation à contre-sens', "Véhicule immobile", "Changement de direction"))
        dict_manv = {'0.0':'Même sens de circulation','1.0':'Circulation à contre-sens', '2.0':"Véhicule immobile", '3.0':"Changement de direction"}
        
        situ = col2.selectbox("Situation de l'accident",
                            ('Sur chaussée', "Sur bande d'arrêt d'urgence", "Sur accotement", "Sur trottoir", "Sur piste cyclable", "Sur autre voie spéciale", "Aucun", "Autre"))
        dict_situ={'1.0':'Sur chaussée', '2.0':"Sur bande d'arrêt d'urgence", '3.0':"Sur accotement", '4.0':"Sur trottoir", '5.0':"Sur piste cyclable", '6.0':"Sur autre voie spéciale", '0.0':"Aucun", '8.0':"Autre"}
        
        #st.write("You selected:", options)

        catr = col2.selectbox("Catégorie de route",
                            ('Autoroute','Route Nationale', 'Route Départementale', "Voie communale", "Hors réseau public", "Parc de stationnement", "Routes de métropole urbaine", "Autre"))
        dict_catr={'1':'Autoroute','2':'Route Nationale', '3':'Route Départementale', '4':"Voie communale",'5':"Hors réseau public",'6': "Parc de stationnement", '7':"Routes de métropole urbaine", '9':"Autre"}
        
        agg = col2.checkbox("L'accident a eu lieu en agglomération. (cochez si oui)")
        inters = col2.checkbox("L'accident a eu lieu dans une intersection. (cochez si oui)")
        
        circ = col2.selectbox("Régime de circulation",('Unidirectionnel','Bidirectionnel'))
        dict_circ = {'0.0':'Unidirectionnel','1.0':'Bidirectionnel'}
        
        prof = col3.selectbox("Profil en long",('Plat','Non plat'))
        dict_prof = {'0.0':'Plat','1.0':'Non plat'}
        
        plan = col3.selectbox("Tracé en plan",('Partie rectiligne','En courbe'))
        dict_plan = {'0.0':'Partie rectiligne','1.0':'En courbe'}
        
        surf = col3.selectbox("Etat de la surface",
                            ('Normale','Mouillée', 'Flaques', "Inondée", "Enneigée", "Présence de boue", "Verglacée", "Grasse, huileuse", "Autre"))
        dict_surf = {'1.0':'Normale','2.0':'Mouillée', '3.0':'Flaques', '4.0':"Inondée", '5.0':"Enneigée", '6.0':"Présence de boue", '7.0':"Verglacée", '8.0':"Grasse, huileuse", '9.0':"Autre"}
        
        infra = col3.selectbox("Aménagement / Infrastructure",
                            ('Aucun','Souterrain-Tunnel', 'Pont', "Bretelle d'échangeur", "Voie ferrée", "Carrefour aménagé", "Zone piétonne", "Zone de péage", "Chantier", "Autre"))
        dict_infra = {'0.0':'Aucun','1.0':'Souterrain-Tunnel', '2.0':'Pont', '3.0':"Bretelle d'échangeur",'4.0':"Voie ferrée", '5.0':"Carrefour aménagé",'6.0': "Zone piétonne", '7.0':"Zone de péage", '8.0':"Chantier", '9.0':"Autre"}
        
        lum = col3.selectbox("Lumière (éclairage)",
                            ("Plein jour", "Crépuscule ou autre", "Nuit sans éclairage (absent ou non allumé)", "Nuit avec éclairage public allumé"))
        dict_lum = {'0.0':"Plein jour", '1.0':"Crépuscule ou autre", '2.0':"Nuit sans éclairage (absent ou non allumé)", '3.0':"Nuit avec éclairage public allumé"}
        
        
        atm = col3.selectbox("Conditions atmosphériques",('Normales',"Dégradées"))
        dict_atm = {'0.0':'Normales','1.0':"Dégradées"}
        
        with st.form(key='information', clear_on_submit=True):
            if st.form_submit_button("**:blue[Prédire l'état de gravité de l'usager]**"):
                try:
                    #Création du dataframe des données
                    data = pd.DataFrame({
                        "mois": d.month,
                        "agg":  ['1' if agg else '0'],
                        "int": ['1.0' if inters else '0.0'],
                        "lat": latitude,
                        "long": longitude,
                        "obs": ['1.0' if obs else '0.0'],
                        "weekend": ['1' if week_end else '0'],
                        "heure": t.hour,
                        "prox_pt_choc": ['1' if prox_pt_choc else '0'],
                        "jour_chome": ['1' if jour_chome else '0'],
                        "age_usager": float(age)
                    })
                    colonnes =  ["lum", "atm", "col", "catr", "circ", "prof", "plan", "surf", "infra", "situ", "sexe", "catv", "obsm", "manv", "motor", "place_rec"]
                    variables =  [lum, atm, col, catr, circ, prof, plan, surf, infra, situ, sexe, catv, obsm, manv, motor, place_rec]
                    for nom, var in zip(colonnes, variables):
                        dictionnaire = f"dict_{nom}"
                        data[nom] = [k  for (k, val) in eval(dictionnaire).items() if val == var]
                    for eq in ['eq_ceinture', 'eq_casque', 'eq_siege', 'eq_gilet', 'eq_airbag', 'eq_gants', 'eq_indetermine', 'eq_autre']:
                        data[eq] = ['1' if dict_eq[eq] in options else '0']
                    data = data.reindex(['mois', 'lum', 'agg', 'int', 'atm', 'col', 'lat', 'long', 'catr','circ', 'prof', 'plan', 'surf', 'infra', 'situ', 'sexe', 'catv', 'obs',
        'obsm', 'manv', 'motor', 'weekend', 'heure', 'place_rec', 'age_usager','eq_ceinture', 'eq_casque', 'eq_siege', 'eq_gilet', 'eq_airbag',
        'eq_gants', 'eq_indetermine', 'eq_autre', 'jour_chome', 'prox_pt_choc'], axis=1)
                    
                    #Preprocessing nécessaire pour le modèle KNN (non intégré)
                    data_preprocessed_knn = preprocessor_knn.transform(data)
  
                    #data_preprocessed = data_preprocessed.reindex(ordre_colonnes, axis=1)
                    
                    #liste_test = '../data/img/' + f'test_knn'
                    #with open(liste_test, "rb") as temp:
                    #    test = pickle.load(temp)
                    #data_test = pd.DataFrame()
                    #for col, val in zip(ordre_colonnes, test):
                    #    data_test.loc[0, col] = val
                    #st.write(data_test)
                    
                    #Preprocessing nécessaire pour le modèle DNN-Keras (non intégré au modèle)
                    data_preprocessed_keras = preprocessor_keras.transform(data)
                    liste_path = '../data/img/' + f'ordre_colonnes_dnn_keras'
                    with open(liste_path, "rb") as temp:
                        ordre_colonnes_keras = pickle.load(temp)
                    data_preprocessed_keras = data_preprocessed_keras.reindex(ordre_colonnes_keras, axis=1)  
                    #st.write(data_preprocessed_keras)
                    
                    #Preprocessing nécessaire pour le modèle DNN-Pytorch(non intégré au modèle)
                    data_preprocessed_pytorch = preprocessor_pytorch.transform(data)
                    data_preprocessed_pytorch = torch.tensor(data_preprocessed_pytorch.astype('float64').values, dtype=torch.float32)
                    
                    #Preprocessing nécessaire pour le modèle Tabnet(non intégré au modèle)
                    data_preprocessed_tabnet = preprocessor_tabnet.transform(data).astype('float64')
                    #st.write(data_preprocessed_tabnet)
                    
                    #Preprocessing nécessaire pour le modèle xgboost(non intégré au modèle)
                    data_preprocessed_xgb = preprocessor_xgb.transform(data).astype('float64')
                    liste_path = '../data/img/' + f'ordre_colonnes_xgboost'
                    with open(liste_path, "rb") as temp:
                        ordre_colonnes_xgb = pickle.load(temp)
                    data_preprocessed_xgb = data_preprocessed_xgb.reindex(ordre_colonnes_xgb, axis=1) 
                    
                    
                    ###Calcul des probabilités d'appartenance avec les différents modèles
                    probs_catboost = catboost_model.predict_proba(data)
                    #st.dataframe(data)
                    probs_logreg = logreg_model.predict_proba(data)
                    cl_svm = svm_model.predict(data)
                    probs_svm = pd.DataFrame([1 if cl_svm==i else 0 for i in range(4)]).T

                    liste_var_rf = ['lat', 'age_usager', 'long', 'heure', 'mois', 'eq_ceinture', 'col','place_rec', 'obsm', 'catv', 'catr', 'eq_casque', 'manv', 'lum',
        'motor', 'obs', 'sexe', 'agg', 'infra', 'weekend', 'jour_chome','prox_pt_choc', 'prof', 'situ', 'circ', 'eq_indetermine', 'surf', 'int',
        'atm', 'plan', 'eq_gants', 'eq_airbag', 'eq_gilet', 'eq_autre']
                    probs_rf = rf_model.predict_proba(data[liste_var_rf])
                    probs_knn = knn_model.predict_proba(data_preprocessed_knn)
                    probs_dnn_keras = dnn_keras_model.predict(data_preprocessed_keras)
                    probs_dnn_pytorch = dnn_pytorch_model(data_preprocessed_pytorch)[0].tolist()
                    probs_tabnet = tabnet_model.predict_proba(data_preprocessed_tabnet.to_numpy())
                    probs_xgb = xgb_model.predict_proba(data_preprocessed_xgb)
                    #st.write(probs_dnn_pytorch)
                    
                    ###Récupération des classes attribuées par les différents modèles
                    cl_logreg = np.argmax(probs_logreg[0])
                    cl_svm = np.argmax(list(probs_svm.values)[0])
                    cl_catboost = np.argmax(list(probs_catboost[0]))
                    cl_rf = np.argmax(list(probs_rf[0]))
                    cl_knn = np.argmax(list(probs_knn[0]))
                    cl_dnn_keras = np.argmax(list(probs_dnn_keras[0]))
                    cl_dnn_pytorch = np.argmax(list(probs_dnn_pytorch))
                    cl_tabnet = np.argmax(list(probs_tabnet))
                    cl_xgb = np.argmax(list(probs_xgb))
                    #st.write(probs_tabnet)
                    
                    st.success(" Avec ces caractéristiques de l'usager et de l'accident, les **niveaux de gravité prédits** sont : ")
                    dict_grav = {0: 'Indemne', 1: 'Blessé léger', 2:'Blessé grave', 3:'Tué'}
                    dict_grav_dnn = {0: 'Indemne', 3: 'Blessé léger', 2:'Blessé grave', 1:'Tué'}
                    classes_predites = pd.DataFrame({
                        "Régression logistique": [dict_grav[cl_logreg]],
                        "LinearSVC + Nystroem" : [dict_grav[cl_svm]],
                        "KNN" : [dict_grav[cl_knn]],
                        "Random Forest": [dict_grav[cl_rf]],
                        "CatBoost": [dict_grav[cl_catboost]],
                        "XGBoost" : [dict_grav[cl_xgb]],
                        "DNN-MLP-Keras" : [dict_grav_dnn[cl_dnn_keras]],
                        "DNN-MLP-Pytorch" : [dict_grav[cl_dnn_pytorch]],
                        "TabNet" : [dict_grav[cl_tabnet]],
                    })
                    st.dataframe(classes_predites, hide_index=True)
                    
                    st.success(" **Probabilités d'appartenance aux différentes classes** ")
                    st.write("""La figure ci-dessous présente les probabilités d'appartenance aux différentes classes, selon le modèle
                            de Machine Learning sélectionné. Le modèle LinearSVC ne permettant pas de calculer des probabilités, une probabilité de 1
                            a été affectée à la classe prédite par le modèle.""")
                    
                    
                    names = ['Indemnes', 'Blessés Légers', 'Blessés Graves', 'Tués']
                    col1, col2, col3 = st.columns([0.1,0.9,0.1])
                    width = 0.1
                    fig, ax = plt.subplots(figsize=(12,6))
                    plt.bar(np.arange(4), list(probs_logreg[0]), width, label="Régression logistique")
                    plt.bar(np.arange(4)+width, list(probs_svm.values)[0], width, label="LinearSVC + Nystrom")
                    plt.bar(np.arange(4)+2*width, list(probs_knn[0]), width, label="KNN")
                    plt.bar(np.arange(4)+3*width, list(probs_rf[0]), width, label="Random Forest")
                    plt.bar(np.arange(4)+4*width, list(probs_catboost[0]), width, label="CatBoost")
                    plt.bar(np.arange(4)+5*width, list(probs_xgb[0]), width, label="XGBoost")
                    plt.bar(np.arange(4)+6*width, [probs_dnn_keras[0][0],probs_dnn_keras[0][3],probs_dnn_keras[0][2],probs_dnn_keras[0][1]], width, label="MLP-Keras")
                    plt.bar(np.arange(4)+7*width, probs_dnn_pytorch, width, label="MLP-Pytorch")
                    plt.bar(np.arange(4)+8*width, list(probs_tabnet[0]), width, label="TabNet")
                    
                    plt.ylabel("Probabilité d'appartenance à la classe")
                    ax.set_xticks(np.arange(4)+4*width, labels = names)
                    plt.legend()
                    col2.pyplot(fig)
                except AttributeError as e:
                    st.write("N'oubliez pas de renseigner la date de l'accident !")
                    #st.write(e)
                
