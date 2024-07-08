import streamlit as st 
import yaml
import pandas as pd
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
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

def onglet_series_temporelles():
    
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sélection des années", "Tendance", "Saisonnalité", "Modélisation"])
    
    with tab1 :
        st.write("##### **:blue[Sélection des années 2021 et 2022 pour les modélisations à cause du biais induit par la covid en 2020]**")
        
        col1, col2 = st.columns([0.5, 0.5]) 
        with col1 :
            st.image('../data/img/covid_indemnes.png', width = 680)
        with col2 :
            st.image('../data/img/covid_blesseslegers.png', width = 680)
    
        st.write("#")
        col1, col2 = st.columns([0.5, 0.5]) 
        with col1 :
            st.image('../data/img/covid_blesseshospitalises.png', width = 680)
        with col2 :
            st.image('../data/img/covid_tues.png', width = 680)
    
    
    with tab2 :
        st.write("##### **:blue[Tendance linéaire à 365 jours]**")

        col1, col2, col3, col4, col5 = st.columns([0.10, 0.35, 0.1, 0.35, 0.10]) 
        with col2 :
            st.write("Courbe des Indemnes lissée avec moyenne mobile")
            st.image('../data/img/Indemnes_365jours.png', width = 500)
        with col4 :
            st.write("Courbe des Blessés légers lissée avec moyenne mobile")
            st.image('../data/img/Blesses_legers_365jours.png', width = 500)
    
        st.write("#")
        col1, col2, col3, col4, col5 = st.columns([0.10, 0.35, 0.1, 0.35, 0.10]) 
        with col2 :
            st.write("Courbe des Blessés hospitalisés lissée avec moyenne mobile")
            st.image('../data/img/Blesses_hospitalises_365jours.png', width = 500)
        with col4 :
            st.write("Courbe des Tués lissée avec moyenne mobile")
            st.image('../data/img/Tues_365jours.png', width = 500)
            

    with tab3 :  
        st.write("##### **:blue[Saisonnalité de 7 jours]**")
        
        col1, col2 = st.columns([0.5, 0.5]) 
        with col1 :
            st.image('../data/img/Indemnes_7jours.png', width = 600)
        with col2 :
            st.image('../data/img/Blesses_legers_7jours.png', width = 600)
    
        st.write("#")
        col1, col2 = st.columns([0.5, 0.5]) 
        with col1 :
            st.image('../data/img/Blesses_hospitalises_7jours.png', width = 600)
        with col2 :
            st.image('../data/img/Tues_7jours.png', width = 600)
            
        st.write("")    
        st.write("De plus, la saisonnalité hebdomadaire est confirmée par les courbes d'autocorrélation et d'autocorrélation partielle.")
        

        st.write("#")         
        st.write("##### **:blue[Confirmation des saisonnalités hebdomadaires et annuelles]**")   
        st.write("Les saisonnalités sont confirmées par les **périodogrammes** :")
        
        def color_coding(row):
            return ['background-color:yellow'] * len(row) if row.Period == 365  or row.Period == 7.02 else ['background-color:white'] * len(row)
                
        
        col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25]) 
        with col1 :
            st.write("Indemnes")
            periodogram_ind = pd.read_csv("../data/img/periodogram_Indemnes.csv")[:5]
            st.dataframe(periodogram_ind.style.apply(color_coding, axis=1), hide_index = True)
        with col2 :
            st.write("Blessés légers")
            periodogram_bl_lg = pd.read_csv("../data/img/periodogram_Blesses_legers.csv")[:5]
            st.dataframe(periodogram_bl_lg.style.apply(color_coding, axis=1), hide_index = True)
        with col3 :
            st.write("Blessés hospitalisés")
            periodogram_bl_hos = pd.read_csv("../data/img/periodogram_Blesses_hospitalises.csv")[:5]
            st.dataframe(periodogram_bl_hos.style.apply(color_coding, axis=1), hide_index = True)
        with col4 :
            st.write("Tués")
            periodogram_tues = pd.read_csv("../data/img/periodogram_Tues.csv")[:5]
            st.dataframe(periodogram_tues.style.apply(color_coding, axis=1), hide_index = True)
        
    
    with tab4 :
        st.write("##### **:blue[Modélisation des séries temporelles]**")
        col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
        modalites = ["Indemnes", "Blessés légers", "Blessés hospitalisés", "Tués"]
        algorithmes = ["Baseline (shift de 1)", 
                      "Baseline (moyenne sur 7 jours + shift de 1)", 
                      "SARIMAX + exog(Fourier)", 
                      "MSTL + AutoARIMA",
                      "MSTL + AutoTheta",
                      "MSTL + AutoCES",
                      "PROPHET", 
                      "PROPHET + vacances scolaires",
                      "PROPHET + jours fériés",
                      "LSTM (look_back de 31 jours)"]
        fits_forecasts = ["Entraînement et évaluation", "Prévisions"]
        st.write("#")
   
    
        with col1 :
            modalite = st.selectbox("Choisissez la gravité des accidentés : ", modalites)
        with col2 :
            algorithme = st.selectbox("Choisissez l'algorithme : ", algorithmes)
        with col3 : 
            fit_forecast = st.selectbox("Choisissez si vous voulez entraîner ou prédire : ", fits_forecasts)
    
    
        if algorithme == 'Baseline (shift de 1)':
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive = pd.read_csv("../data/saved_models/Indemnes_Baseline_shift1.csv", index_col = 0)
                    #st.dataframe(df_naive.head())
                
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive = pd.read_csv("../data/saved_models/Blesses_legers_Baseline_shift1.csv", index_col = 0)
                    #st.dataframe(df_naive.head())
                
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive = pd.read_csv("../data/saved_models/Blesses_hospitalises_Baseline_shift1.csv", index_col = 0)
                    #st.dataframe(df_naive.head())
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive = pd.read_csv("../data/saved_models/Tues_Baseline_shift1.csv", index_col = 0)
                    #st.dataframe(df_naive.head())
            
                col1, col2 = st.columns([0.5, 0.5])
            
                with col1 :
                    st.write(':blue[Courbe sur toute la période]')
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(df_naive.index, df_naive['Nbre_Acc'], alpha = 0.8, label = 'True')
                    plt.plot(df_naive.index, df_naive['Nbre_Acc_t-1'], alpha = 0.8, label = 'Shift 1', linestyle = "dashed")
                    plt.legend()
                    plt.xlim(df_naive.index[0], df_naive.index[-1])
                    plt.xticks([0, (365-1), (2*365-1)], ['02-01-2021', '01-01-2022', '31-12-2022'])
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"Baseline à j-1 pour les {modalite}")
                    st.pyplot(fig)
            
            
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :  
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                    true = df_naive['Nbre_Acc']
                    prediction = df_naive['Nbre_Acc_t-1']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(true, prediction)
                    train_mse = mean_squared_error(true, prediction)
                    train_rmse = root_mean_squared_error(true, prediction)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        })

                    st.dataframe(performance_df)
                    
            if fit_forecast == "Prévisions" :
                st.write(":red[Pas de prévision possible avec cet algorithme]")
            
            
        elif algorithme == 'Baseline (moyenne sur 7 jours + shift de 1)':
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive2 = pd.read_csv("../data/saved_models/Indemnes_Baseline_moy7+shift1.csv", index_col = 0)
                    #st.dataframe(df_naive2.head())
                
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive2 = pd.read_csv("../data/saved_models/Blesses_legers_Baseline_moy7+shift1.csv", index_col = 0)
                    #st.dataframe(df_naive2.head())
                
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive2 = pd.read_csv("../data/saved_models/Blesses_hospitalises_Baseline_moy7+shift1.csv", index_col = 0)
                    #st.dataframe(df_naive2.head())
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_naive2 = pd.read_csv("../data/saved_models/Tues_Baseline_moy7+shift1.csv", index_col = 0)
                    #st.dataframe(df_naive2.head())
            
            
                col1, col2 = st.columns([0.5, 0.5])
            
                with col1 :
                    st.write(':blue[Courbe sur toute la période]')
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(df_naive2.index, df_naive2['Nbre_Acc'], alpha = 0.8, label = 'True')
                    plt.plot(df_naive2.index, df_naive2['Nbre_Acc_t-1'], alpha = 0.8, label = 'Shift 1', linestyle = "dashed")
                    plt.legend()
                    plt.xlim(df_naive2.index[0], df_naive2.index[-1])
                    plt.xticks([0, (365-8), (2*365-8)], ['08-01-2021', '01-01-2022', '31-12-2022'])
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"Baseline à j-1 pour les {modalite}")
                    st.pyplot(fig)
            
            
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :  
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                    true = df_naive2['Nbre_Acc']
                    prediction = df_naive2['Nbre_Acc_t-1']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(true, prediction)
                    train_mse = mean_squared_error(true, prediction)
                    train_rmse = root_mean_squared_error(true, prediction)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        })

                    st.dataframe(performance_df)
                    
            if fit_forecast == "Prévisions" :
                st.write(":red[Pas de prévision possible avec cet algorithme]")
            
            
        elif algorithme == 'SARIMAX + exog(Fourier)':
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_sarimax = pd.read_csv("../data/saved_models/Indemnes_Sarimax.csv", index_col = 0)
                    sarimax = joblib.load("../data/saved_models/Indemnes_Sarimax.joblib")
                    #st.dataframe(df_sarimax.head())
                
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_sarimax = pd.read_csv("../data/saved_models/Blesses_legers_Sarimax.csv", index_col = 0)
                    sarimax = joblib.load("../data/saved_models/Blesses_legers_Sarimax.joblib")
                    #st.dataframe(df_sarimax.head())
                
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_sarimax = pd.read_csv("../data/saved_models/Blesses_hospitalises_Sarimax.csv", index_col = 0)
                    sarimax = joblib.load("../data/saved_models/Blesses_hospitalises_Sarimax.joblib")
                    #st.dataframe(df_sarimax.head())
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    df_sarimax = pd.read_csv("../data/saved_models/Tues_Sarimax.csv", index_col = 0)
                    sarimax = joblib.load("../data/saved_models/Tues_Sarimax.joblib")
                    #st.dataframe(df_sarimax.head())
                
                
                # Séparation en train et test
                train, test = train_test_split(df_sarimax, test_size = 0.1, shuffle = False)
                
                x = train['Nbre_Acc'].values
                
                
                # Extrapolation de la série de Fourier
                def fourierExtrapolation(x, n_predict):
                    n = x.size
    
                    n_harm = 20                     # number of harmonics in model
    
                    t = np.arange(0, n)
                    p = np.polyfit(t, x, 1)         # find linear trend in x
    
                    x_notrend = x - p[0] * t        # detrended x
    
                    x_freqdom = fft(x_notrend)  # detrended x in frequency domain
                    f = fftfreq(n)              # frequencies
    
                    indexes = list(range(n))
                    # sort indexes by frequency, lower -> higher
                    indexes.sort(key = lambda i: np.absolute(f[i]))
 
                    t = np.arange(0, n + n_predict)
                    restored_sig = np.zeros(t.size)
    
                    for i in indexes[:1 + n_harm * 2]:
                        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
                        phase = np.angle(x_freqdom[i])          # phase
                        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    
                    return restored_sig + p[0] * t

                exog = fourierExtrapolation(x, len(test))
                
                train['exog'] = exog[:len(train)]
                test['exog'] = exog[len(train):]
            
            
                col1, col2 = st.columns([0.5, 0.5])
            
                with col1 :
                    st.write(':blue[Courbe sur toute la période]')
                    
                    fig, ax = plt.subplots(figsize = (15, 5))
                    plt.plot(train.index, train['Nbre_Acc'], label = 'Train')
                    plt.plot(test.index, test['Nbre_Acc'], color = 'orange',label = 'Test')
                    
                    plt.plot(train.index, train['exog'], 'b-',label = 'Série de Fourier pour train')
                    plt.plot(test.index, test['exog'], 'r-', label = 'Série de Fourire pour test')
                    
                    fcast = sarimax.get_forecast(len(test), exog= test['exog']).summary_frame()
                    plt.plot(test.index, fcast['mean'], 'k--', label = 'Predictions SARIMAX')
                    
                    ax.fill_between(test.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1, label = 'Déviation standard')
                    
                    plt.xticks([0, 365, (2*365-1)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                    plt.xlim(train.index[0], test.index[-1])
                    
                    plt.title(f"SARIMAX avec série de Fourier pour les {modalite}")
                    plt.legend()
                    plt.xlabel("Dates")
                    plt.ylabel(f"Nombre de {modalite}")
                    st.pyplot(fig)
                    
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                    
                    fig, ax = plt.subplots(figsize = (15, 5))
                    plt.plot(train.index[-110 :], train['Nbre_Acc'][-110 :], label = 'Train')
                    plt.plot(test.index, test['Nbre_Acc'], color = 'orange',label = 'Test')
                    
                    plt.plot(train.index[-110 :], train['exog'][-110 :], 'b-',label = 'Série de Fourier pour train')
                    plt.plot(test.index, test['exog'], 'r-', label = 'Série de Fourire pour test')
                    
                    fcast = sarimax.get_forecast(len(test), exog= test['exog']).summary_frame()
                    plt.plot(test.index, fcast['mean'], 'k--', label = 'Predictions SARIMAX')
                    
                    ax.fill_between(test.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1, label = 'Déviation standard')
                    
                    plt.xticks([0, 31, 62, 92, 123, 153, 183], ['01-07-2021', '01-08-2021', '01-09-2022', '01-10-2022','01-11-2022','01-12-2022','31-12-2022'])
                    plt.xlim(train.index[-110], test.index[-1])
                    
                    plt.title(f"Agrandissement de la partie droite de SARIMAX avec série de Fourier pour les {modalite}")
                    plt.legend()
                    plt.xlabel("Dates")
                    plt.ylabel(f"Nombre de {modalite}")
                    st.pyplot(fig)
            
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :  
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 

                    train_predictions = sarimax.get_forecast(len(train), exog= train['exog']).summary_frame()['mean']
                    test_predictions = sarimax.get_forecast(len(test), exog= test['exog']).summary_frame()['mean']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train['Nbre_Acc'], train_predictions)
                    train_mse = mean_squared_error(train['Nbre_Acc'], train_predictions)
                    train_rmse = root_mean_squared_error(train['Nbre_Acc'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test['Nbre_Acc'], test_predictions)
                    test_mse = mean_squared_error(test['Nbre_Acc'], test_predictions)
                    test_rmse = root_mean_squared_error(test['Nbre_Acc'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                    
            if fit_forecast == "Prévisions" :
                st.write(":red[Pas de prévision possible avec cet algorithme]")
       
            
        elif algorithme == 'MSTL + AutoARIMA':
            if modalite == "Indemnes" :
                df_MSTL = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                
            elif modalite == "Blessés légers" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                
            elif modalite == "Blessés hospitalisés" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                
            elif modalite == "Tués" :
                df_MSTL = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
        
                    
            # Séparation en train et test
            train_MSTL, test_MSTL = train_test_split(df_MSTL, test_size = 0.1, shuffle = False)
                    
            horizon_MSTL = len(test_MSTL)
                    
            models = [MSTL(season_length=[7, 7 * 52], 
                          trend_forecaster=AutoARIMA(prediction_intervals=ConformalIntervals(n_windows=3, h=horizon_MSTL)))]
                    
            sf = StatsForecast(models = models,
                              freq = 'D',     
                              n_jobs = -1)
                    
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
            
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
            
                sf.fit(df = train_MSTL)
                    
                horizon = len(test_MSTL)
                Y_test_pred = sf.forecast(horizon, fitted=True)
                Y_train_pred = sf.forecast_fitted_values()
                Y_test_pred = Y_test_pred.reset_index()
                    
                    
                col1, col2 = st.columns([0.5, 0.5])
            
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                        
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(Y_train_pred['ds'], Y_train_pred['y'], alpha = 0.8, label = 'Train true')
                    plt.plot(Y_train_pred['ds'], Y_train_pred['MSTL'], alpha = 0.8, label = 'Train MSTL', linestyle = "dashed")
                    plt.plot(Y_test_pred['ds'], test_MSTL['y'], alpha = 0.8, label = 'Test true')
                    plt.plot(Y_test_pred['ds'], Y_test_pred['MSTL'], alpha = 0.8, label = 'Test MSTL', linestyle = "dashed")
                    plt.legend()
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[0], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                        
                    fcast = sf.forecast(h=horizon, level=[95])
                        
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(Y_train_pred['ds'][-110 :], Y_train_pred['y'][-110 :], label = 'Train true')
                    ax.plot(Y_test_pred['ds'], test_MSTL['y'], label = 'Test true')
                    ax.plot(Y_test_pred['ds'], fcast['MSTL'], "k--",label = 'Test MSTL', alpha = 0.6)
                    ax.fill_between(Y_test_pred['ds'], fcast['MSTL-lo-95'], fcast['MSTL-hi-95'], color='k', alpha=0.1)
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    ax.set_title(f"Agrandissement de la partie droite de MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[-110], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)

                    
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                        
                    train_predictions = Y_train_pred['MSTL']
                    test_predictions = Y_test_pred['MSTL']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train_MSTL['y'], train_predictions)
                    train_mse = mean_squared_error(train_MSTL['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train_MSTL['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test_MSTL['y'], test_predictions)
                    test_mse = mean_squared_error(test_MSTL['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test_MSTL['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                        })

                    st.dataframe(performance_df)
                        
                        
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                sf.fit(df = df_MSTL)      
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                    options = [30, 60, 90, 120, 150, 180])
                
                
                Y_pred = sf.forecast(horizon, fitted=True)
                Y_pred['ds'] = Y_pred['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_MSTL['ds'], df_MSTL['y'], label = 'Jeu de données')
                plt.plot(Y_pred['ds'], Y_pred['MSTL'], label = 'Prédictions MSTL')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_MSTL['ds'].iloc[0], Y_pred['ds'].iloc[-1])
                st.pyplot(fig)               
                        
            
        elif algorithme == 'MSTL + AutoTheta':
            if modalite == "Indemnes" :
                df_MSTL = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés légers" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Tués" :
                df_MSTL = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            # Séparation en train et test
            train_MSTL, test_MSTL = train_test_split(df_MSTL, test_size = 0.1, shuffle = False)
                        
            horizon_MSTL = len(test_MSTL)          

            models = [MSTL(season_length=[7, 7 * 52], 
                            trend_forecaster=AutoTheta(prediction_intervals=ConformalIntervals(n_windows=3, h=horizon_MSTL)))]
                        
            sf = StatsForecast(models = models,
                                freq = 'D',     
                                n_jobs = -1)
                    
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)

                sf.fit(df = train_MSTL)
                    
                horizon = len(test_MSTL)
                Y_test_pred = sf.forecast(horizon, fitted=True)
                Y_train_pred = sf.forecast_fitted_values()
                Y_test_pred = Y_test_pred.reset_index()
                        
                        
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                            
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(Y_train_pred['ds'], Y_train_pred['y'], alpha = 0.8, label = 'Train true')
                    plt.plot(Y_train_pred['ds'], Y_train_pred['MSTL'], alpha = 0.8, label = 'Train MSTL', linestyle = "dashed")
                    plt.plot(Y_test_pred['ds'], test_MSTL['y'], alpha = 0.8, label = 'Test true')
                    plt.plot(Y_test_pred['ds'], Y_test_pred['MSTL'], alpha = 0.8, label = 'Test MSTL', linestyle = "dashed")
                    plt.legend()
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[0], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                            
                    fcast = sf.forecast(h=horizon, level=[95])
                            
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(Y_train_pred['ds'][-110 :], Y_train_pred['y'][-110 :], label = 'Train true')
                    ax.plot(Y_test_pred['ds'], test_MSTL['y'], label = 'Test true')
                    ax.plot(Y_test_pred['ds'], fcast['MSTL'], "k--",label = 'Test MSTL', alpha = 0.6)
                    ax.fill_between(Y_test_pred['ds'], fcast['MSTL-lo-95'], fcast['MSTL-hi-95'], color='k', alpha=0.1)
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    ax.set_title(f"Agrandissement de la partie droite de MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[-110], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)

                        
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                            
                    train_predictions = Y_train_pred['MSTL']
                    test_predictions = Y_test_pred['MSTL']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train_MSTL['y'], train_predictions)
                    train_mse = mean_squared_error(train_MSTL['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train_MSTL['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test_MSTL['y'], test_predictions)
                    test_mse = mean_squared_error(test_MSTL['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test_MSTL['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                        
                        
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                sf.fit(df = df_MSTL)      
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                options = [30, 60, 90, 120, 150, 180])
                
                
                Y_pred = sf.forecast(horizon, fitted=True)
                Y_pred['ds'] = Y_pred['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_MSTL['ds'], df_MSTL['y'], label = 'Jeu de données')
                plt.plot(Y_pred['ds'], Y_pred['MSTL'], label = 'Prédictions MSTL')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_MSTL['ds'].iloc[0], Y_pred['ds'].iloc[-1])
                st.pyplot(fig)
                
            
        elif algorithme == 'MSTL + AutoCES':
            if modalite == "Indemnes" :
                df_MSTL = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés légers" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_MSTL = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Tués" :
                df_MSTL = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                        
            # Séparation en train et test
            train_MSTL, test_MSTL = train_test_split(df_MSTL, test_size = 0.1, shuffle = False)
                        
            # Essai du modèle directement
            horizon_MSTL = len(test_MSTL)
                    
            models = [MSTL(season_length=[7, 7 * 52], 
                            trend_forecaster=AutoCES(prediction_intervals=ConformalIntervals(n_windows=3, h=horizon_MSTL)))]
                        
            sf = StatsForecast(models = models,
                                    freq = 'D',     
                                    n_jobs = -1)
                        
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)

                sf.fit(df = train_MSTL)
                        
                horizon = len(test_MSTL)
                Y_test_pred = sf.forecast(horizon, fitted=True)
                Y_train_pred = sf.forecast_fitted_values()
                Y_test_pred = Y_test_pred.reset_index()
                    
                    
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                            
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(Y_train_pred['ds'], Y_train_pred['y'], alpha = 0.8, label = 'Train true')
                    plt.plot(Y_train_pred['ds'], Y_train_pred['MSTL'], alpha = 0.8, label = 'Train MSTL', linestyle = "dashed")
                    plt.plot(Y_test_pred['ds'], test_MSTL['y'], alpha = 0.8, label = 'Test true')
                    plt.plot(Y_test_pred['ds'], Y_test_pred['MSTL'], alpha = 0.8, label = 'Test MSTL', linestyle = "dashed")
                    plt.legend()
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[0], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                            
                    fcast = sf.forecast(h=horizon, level=[95])
                            
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(Y_train_pred['ds'][-110 :], Y_train_pred['y'][-110 :], label = 'Train true')
                    ax.plot(Y_test_pred['ds'], test_MSTL['y'], label = 'Test true')
                    ax.plot(Y_test_pred['ds'], fcast['MSTL'], "k--",label = 'Test MSTL', alpha = 0.6)
                    ax.fill_between(Y_test_pred['ds'], fcast['MSTL-lo-95'], fcast['MSTL-hi-95'], color='k', alpha=0.1)
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    ax.set_title(f"Agrandissement de la partie droite de MSTL forecast pour les {modalite}")
                    plt.xlim(Y_train_pred['ds'].iloc[-110], Y_test_pred['ds'].iloc[-1])
                    st.pyplot(fig)

                    
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
                
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                            
                    train_predictions = Y_train_pred['MSTL']
                    test_predictions = Y_test_pred['MSTL']

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train_MSTL['y'], train_predictions)
                    train_mse = mean_squared_error(train_MSTL['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train_MSTL['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test_MSTL['y'], test_predictions)
                    test_mse = mean_squared_error(test_MSTL['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test_MSTL['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                        
                        
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    
                sf.fit(df = df_MSTL)      
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                options = [30, 60, 90, 120, 150, 180])
                
                
                Y_pred = sf.forecast(horizon, fitted=True)
                Y_pred['ds'] = Y_pred['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_MSTL['ds'], df_MSTL['y'], label = 'Jeu de données')
                plt.plot(Y_pred['ds'], Y_pred['MSTL'], label = 'Prédictions MSTL')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_MSTL['ds'].iloc[0], Y_pred['ds'].iloc[-1])
                st.pyplot(fig)
            
            
        elif algorithme == 'PROPHET':
            if modalite == "Indemnes" :
                df_PROPHET = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés légers" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Tués" :
                df_PROPHET = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                        
                        
            # Séparation en train et test
            train, test = train_test_split(df_PROPHET, test_size = 0.1, shuffle = False)
                    
        
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_train.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_train.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_train.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)  
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_train.joblib")      
                    
                    
                test_dates = my_model.make_future_dataframe(periods = len(test), freq = 'D')
                forecast = my_model.predict(test_dates)
                #st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
                        
                col1, col2 = st.columns([0.5, 0.5])
            
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'], train['y'], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'], forecast['yhat'], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='k', alpha=0.1)
                    ax.set_title(f"PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[0], test['ds'].iloc[-1])
                    plt.xticks([0, (365), (2*365)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                    st.pyplot(fig)       
                        
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'][-110 :], train['y'][-110 :], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat'][-(110 + len(test)) :], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat_lower'][-(110 + len(test)) :], forecast['yhat_upper'][-(110 + len(test)) :], color='k', alpha=0.1)
                    ax.set_title(f"Agrandissement de la partie droite de PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[-110], test['ds'].iloc[-1])
                    plt.xticks([0, 31, 62, 92, 123, 153, 183], ['01-07-2021', '01-08-2021', '01-09-2022', '01-10-2022','01-11-2022','01-12-2022','31-12-2022'])
                    st.pyplot(fig)  
                        
                    
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
                
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                    
                    train_predictions = forecast['yhat'][ : len(train)]
                    test_predictions = forecast['yhat'][len(train) : ]

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train['y'], train_predictions)
                    train_mse = mean_squared_error(train['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test['y'], test_predictions)
                    test_mse = mean_squared_error(test['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                    
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_df.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_df.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_df.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_df.joblib")
                  

             
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                options = [30, 60, 90, 120, 150, 180])
                
                futur_dates = my_model.make_future_dataframe(periods = horizon, freq = 'D')
                forecast = my_model.predict(futur_dates)
                forecast['ds'] = forecast['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_PROPHET['ds'], df_PROPHET['y'], label = 'Jeu de données')
                plt.plot(forecast['ds'][-horizon : ], forecast['yhat'][-horizon : ], label = 'Prédictions PROPHET')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_PROPHET['ds'].iloc[0], forecast['ds'].iloc[-1])
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                st.pyplot(fig)                   
            
            
        elif algorithme == 'PROPHET + vacances scolaires':
            if modalite == "Indemnes" :
                df_PROPHET = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés légers" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Tués" :
                df_PROPHET = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
                    
            # Séparation en train et test
            train, test = train_test_split(df_PROPHET, test_size = 0.1, shuffle = False)
            
            
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_vacances_train.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_vacances_train.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_vacances_train.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_vacances_train.joblib")       
                        
                        
                test_dates = my_model.make_future_dataframe(periods = len(test), freq = 'D')
                forecast = my_model.predict(test_dates)
                #st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
                    
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'], train['y'], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'], forecast['yhat'], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='k', alpha=0.1)
                    ax.set_title(f"PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[0], test['ds'].iloc[-1])
                    plt.xticks([0, (365), (2*365)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                    st.pyplot(fig)       
                        
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'][-110 :], train['y'][-110 :], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat'][-(110 + len(test)) :], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat_lower'][-(110 + len(test)) :], forecast['yhat_upper'][-(110 + len(test)) :], color='k', alpha=0.1)
                    ax.set_title(f"Agrandissement de la partie droite de PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[-110], test['ds'].iloc[-1])
                    plt.xticks([0, 31, 62, 92, 123, 153, 183], ['01-07-2021', '01-08-2021', '01-09-2022', '01-10-2022','01-11-2022','01-12-2022','31-12-2022'])
                    st.pyplot(fig)  
                            
                        
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                    
                    train_predictions = forecast['yhat'][ : len(train)]
                    test_predictions = forecast['yhat'][len(train) : ]

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train['y'], train_predictions)
                    train_mse = mean_squared_error(train['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test['y'], test_predictions)
                    test_mse = mean_squared_error(test['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_vacances_df.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_vacances_df.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_vacances_df.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_vacances_df.joblib")  

                
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                options = [30, 60, 90, 120, 150, 180])
            
                futur_dates = my_model.make_future_dataframe(periods = horizon, freq = 'D')
                forecast = my_model.predict(futur_dates)
                forecast['ds'] = forecast['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_PROPHET['ds'], df_PROPHET['y'], label = 'Jeu de données')
                plt.plot(forecast['ds'][-horizon : ], forecast['yhat'][-horizon : ], label = 'Prédictions PROPHET')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_PROPHET['ds'].iloc[0], forecast['ds'].iloc[-1])
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                st.pyplot(fig)                   
                
            
        elif algorithme == 'PROPHET + jours fériés':
            if modalite == "Indemnes" :
                df_PROPHET = pd.read_csv("../data/saved_models/Indemnes_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés légers" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_legers_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_PROPHET = pd.read_csv("../data/saved_models/Blesses_hospitalises_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                    
            elif modalite == "Tués" :
                df_PROPHET = pd.read_csv("../data/saved_models/Tues_MSTL.csv", index_col = 0)
                #st.dataframe(df_MSTL.head())
                        
                        
            # Séparation en train et test
            train, test = train_test_split(df_PROPHET, test_size = 0.1, shuffle = False)
        
        
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_feries_train.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_feries_train.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_feries_train.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_feries_train.joblib")        
                        
                        
                test_dates = my_model.make_future_dataframe(periods = len(test), freq = 'D')
                forecast = my_model.predict(test_dates)
                #st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
                    
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'], train['y'], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'], forecast['yhat'], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='k', alpha=0.1)
                    ax.set_title(f"PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[0], test['ds'].iloc[-1])
                    plt.xticks([0, (365), (2*365)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                    st.pyplot(fig)       
                        
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(train['ds'][-110 :], train['y'][-110 :], label = 'Train true')
                    ax.plot(test['ds'], test['y'], label = 'Test true')
                    ax.plot(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat'][-(110 + len(test)) :], "k--",label = 'Train Prophet', alpha = 0.6)
                    ax.fill_between(df_PROPHET['ds'][-(110 + len(test)) :], forecast['yhat_lower'][-(110 + len(test)) :], forecast['yhat_upper'][-(110 + len(test)) :], color='k', alpha=0.1)
                    ax.set_title(f"Agrandissement de la partie droite de PROPHET forecast pour les {modalite}")
                    ax.set_xlabel('Dates')
                    ax.set_ylabel(f"Nombre de {modalite}")
                    plt.legend()
                    plt.xlim(train['ds'].iloc[-110], test['ds'].iloc[-1])
                    plt.xticks([0, 31, 62, 92, 123, 153, 183], ['01-07-2021', '01-08-2021', '01-09-2022', '01-10-2022','01-11-2022','01-12-2022','31-12-2022'])
                    st.pyplot(fig)  
                            
                        
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 
                    
                    train_predictions = forecast['yhat'][ : len(train)]
                    test_predictions = forecast['yhat'][len(train) : ]

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(train['y'], train_predictions)
                    train_mse = mean_squared_error(train['y'], train_predictions)
                    train_rmse = root_mean_squared_error(train['y'], train_predictions)

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(test['y'], test_predictions)
                    test_mse = mean_squared_error(test['y'], test_predictions)
                    test_rmse = root_mean_squared_error(test['y'], test_predictions)

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)
                
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Indemnes_PROPHET_feries_df.joblib")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_legers_PROPHET_feries_df.joblib")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Blesses_hospitalises_PROPHET_feries_df.joblib")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    my_model = joblib.load("../data/saved_models/Tues_PROPHET_feries_df.joblib")

                
                col1, col2 = st.columns([0.5, 0.5])
                with col1 :
                    horizon = st.select_slider("Choisissez la nombre de jours de prédiction",
                                                options = [30, 60, 90, 120, 150, 180])
            
                futur_dates = my_model.make_future_dataframe(periods = horizon, freq = 'D')
                forecast = my_model.predict(futur_dates)
                forecast['ds'] = forecast['ds'].astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_PROPHET['ds'], df_PROPHET['y'], label = 'Jeu de données')
                plt.plot(forecast['ds'][-horizon : ], forecast['yhat'][-horizon : ], label = 'Prédictions PROPHET')
                plt.legend()
                plt.xlabel('Dates')
                plt.ylabel(f"Nombre de {modalite}")
                plt.xticks([0, 365, (2*365)], ['01-01-2021', '01-01-2022', '01-01-2023'])
                plt.xlim(df_PROPHET['ds'].iloc[0], forecast['ds'].iloc[-1])
                plt.title(f"Prédiction pour les {horizon} prochains jours de 2023 pour les {modalite}")
                st.pyplot(fig)                   
            
            
        elif algorithme == 'LSTM (look_back de 31 jours)':
            if modalite == "Indemnes" :
                df_LSTM = pd.read_csv("../data/saved_models/Indemnes_Sarimax.csv", index_col = 0)
                #st.dataframe(df_LSTM.head())
                    
            elif modalite == "Blessés légers" :
                df_LSTM = pd.read_csv("../data/saved_models/Blesses_legers_Sarimax.csv", index_col = 0)
                #st.dataframe(df_LSTM.head())
                    
            elif modalite == "Blessés hospitalisés" :
                df_LSTM = pd.read_csv("../data/saved_models/Blesses_hospitalises_Sarimax.csv", index_col = 0)
                #st.dataframe(df_LSTM.head())
                    
            elif modalite == "Tués" :
                df_LSTM = pd.read_csv("../data/saved_models/Tues_Sarimax.csv", index_col = 0)
                #st.dataframe(df_LSTM.head())
            
            dataset = df_LSTM.values
            
            # nNormalisation du dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset_scaled = scaler.fit_transform(dataset)         
                        
            # Séparation en train et test
            train_scaled, test_scaled = train_test_split(dataset_scaled, test_size = 0.1, shuffle = False)
        
            # Conversion d'un array en matrice
            def create_dataset(dataset, look_back=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-look_back-1):
                    dataX.append(dataset[i:(i+look_back), 0])
                    dataY.append(dataset[i + look_back, 0])
                return np.array(dataX), np.array(dataY)
            
            # Redimensionnement en X=t and Y=t+1
            look_back = 31
            trainX_scaled, trainY_scaled = create_dataset(train_scaled, look_back)
            testX_scaled, testY_scaled = create_dataset(test_scaled, look_back)
            
            # Redimensionnemtn de l'input pour être de la forme [samples, time steps, features]
            trainX_scaled= np.reshape(trainX_scaled, (trainX_scaled.shape[0], 1, trainX_scaled.shape[1]))
            testX_scaled = np.reshape(testX_scaled, (testX_scaled.shape[0], 1, testX_scaled.shape[1]))
            #st.dataframe(trainY_scaled)
        
        
            if fit_forecast == "Entraînement et évaluation" :
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Indemnes_LSTM.h5")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Blesses_legers_LSTM.h5")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Blesses_hospitalises_LSTM.h5")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez entraîner la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Tues_LSTM.h5")       
                            
                trainPredict = model.predict(trainX_scaled)
                testPredict = model.predict(testX_scaled)

                # invert predictions
                trainPredict = scaler.inverse_transform(trainPredict)
                trainY = scaler.inverse_transform([trainY_scaled])
                testPredict = scaler.inverse_transform(testPredict)
                testY = scaler.inverse_transform([testY_scaled])
                    
                    
                col1, col2 = st.columns([0.5, 0.5])
                
                with col1 :
                    st.write(':blue[Courbe sur toute la période]') 
                    
                    # shift des prédictions de train pour la visaulisation
                    trainPredictPlot = np.empty_like(dataset_scaled)
                    trainPredictPlot[:, :] = np.nan
                    trainPredictPlot[look_back : len(trainPredict)+look_back, :] = trainPredict

                    # shift des prédictions de test pour la visaulisation
                    testPredictPlot = np.empty_like(dataset_scaled)
                    testPredictPlot[:, :] = np.nan
                    testPredictPlot[len(trainPredict)+(look_back*2)+1 : len(dataset_scaled)-1, :] = testPredict
    
                    
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(df_LSTM.index, scaler.inverse_transform(dataset_scaled), label = 'True', alpha = 0.6)
                    plt.plot(df_LSTM.index, trainPredictPlot, "--", label = 'train predict')
                    plt.plot(df_LSTM.index, testPredictPlot, "--", label = 'test predict')
                    plt.legend()
                    plt.xticks([0, 365, (2*365-1)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                    plt.xlim(df_LSTM.index[0], df_LSTM.index[-1])
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"LSTM pour les {modalite}")
                    st.pyplot(fig)
                              
            
                with col2 :
                    st.write(':blue[Agrandissement de la partie droite de la courbe]')
                    
                    fig = plt.figure(figsize = (15, 5))
                    plt.plot(df_LSTM.index[-110 :], scaler.inverse_transform(dataset_scaled)[-110 :], label = 'True', alpha = 0.6)
                    plt.plot(df_LSTM.index[-110 :], trainPredictPlot[-110 :], "--", label = 'train predict')
                    plt.plot(df_LSTM.index[-110 :], testPredictPlot[-110 :], "--", label = 'test predict')
                    plt.legend()
                    plt.xticks([0, 31, 62, 92, 123, 153, 183], ['01-07-2021', '01-08-2021', '01-09-2022', '01-10-2022','01-11-2022','01-12-2022','31-12-2022'])
                    plt.xlim(df_LSTM.index[-110], df_LSTM.index[-1])
                    plt.xlabel('Dates')
                    plt.ylabel(f"Nombre de {modalite}")
                    plt.title(f"grandissement de la partie droite de LSTM pour les {modalite}")
                    st.pyplot(fig)
                    
                    
                col1, col2, col3 = st.columns([ 0.3, 0.45, 0.25])
            
                with col2 :    
                    st.write("#")
                    st.write(':blue[Évaluation du modèle selon 3 métriques]') 

                    # Mesures de performance sur l'ensemble d'entraînement
                    train_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
                    train_mse = mean_squared_error(trainY[0], trainPredict[:,0])
                    train_rmse = root_mean_squared_error(trainY[0], trainPredict[:,0])

                    # Mesures de performance sur l'ensemble de test
                    test_mae = mean_absolute_error(testY[0], testPredict[:,0])
                    test_mse = mean_squared_error(testY[0], testPredict[:,0])
                    test_rmse = root_mean_squared_error(testY[0], testPredict[:,0])

                    # Créer un DataFrame pour afficher les mesures de performance
                    performance_df = pd.DataFrame({
                        'Métrique': ['MAE', 'MSE', 'RMSE'],
                        'Ensemble d\'entraînement': [train_mae, train_mse, train_rmse],
                        'Ensemble de test': [test_mae, test_mse, test_rmse]
                    })

                    st.dataframe(performance_df)       
    
    
            if fit_forecast == "Prévisions" : 
                if modalite == "Indemnes" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Indemnes_LSTM.h5")
                    
                elif modalite == "Blessés légers" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Blesses_legers_LSTM.h5")
                    
                elif modalite == "Blessés hospitalisés" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Blesses_hospitalises_LSTM.h5")
                
                elif modalite == "Tués" :
                    st.write('Vous souhaitez prédire la modalité', modalite, " avec l'algorithme ", algorithme)
                    model  = load_model("../data/saved_models/Tues_LSTM.h5")

                
                sample = np.reshape(testX_scaled[-1], (1, 1, look_back))
                pred_scaled = []
                n = look_back                 
                for i in range(n):
                    next_step = model.predict(sample)
                    add_sample = np.append(sample, next_step)[1:]
                    sample = np.reshape(add_sample, (1, 1, add_sample.shape[0]))
                    pred_scaled.append(next_step[0,0])
                        
                pred_scaled = np.array(pred_scaled)
                pred_scaled = pred_scaled.reshape(len(pred_scaled), 1)
                    
                pred = scaler.inverse_transform(pred_scaled)
                    
                start = datetime.datetime.strptime("2023-01-01", "%Y-%m-%d")
                date_generate = pd.date_range(start, periods = n)
                date_generate = date_generate.astype(str)
                    
                fig = plt.figure(figsize = (15, 5))
                plt.plot(df_LSTM.index, scaler.inverse_transform(dataset_scaled), label = 'True')
                plt.plot(date_generate, pred, label = 'prédictions')
                plt.xticks([0, 365, (2*365-1)], ['01-01-2021', '01-01-2022', '31-12-2022'])
                plt.xlim(0,(2*365-1+look_back))
                plt.title(f"Prédictions pour le premier mois de 2023 pour les {modalite}")
                plt.legend()
                plt.xlabel("Dates")
                plt.ylabel(f"Nombre de {modalite}")
                st.pyplot(fig)         

