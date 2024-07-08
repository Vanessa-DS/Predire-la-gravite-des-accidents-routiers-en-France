import streamlit as st 
import yaml
import pandas as pd
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import joblib

import datetime

def onglet_accueil():
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
    with col1:
        st.markdown("* [Matthieu Claudel](http://www.linkedin.com/in/matthieu-claudel-8a927857) \
                    \n * [Vanessa Ibert](http://www.linkedin.com/in/vanessa-ibert) \
                    \n * [Camille Pelat](http://www.linkedin.com/in/camille-pelat-08a7b68a) \
                    \n * [Nadège Reboul](http://www.linkedin.com/in/nadege-reboul)")
        st.write("Tutorat : Axalia Levenchaud")
    with col3:
        st.image('./assets/logo-datascientest.png')

    col1, col2 = st.columns([0.3, 0.7])
    with col2 :
        st.image('../data/img/Photo_accueil.png', width = 400)
    
    col1, col2, col3 = st.columns([0.23, 0.07, 0.70])        
    with col2 :
        st.image('./assets/logo-GitHub.png', width = 50) #../data/img/logo-GitHub.png , width = 50
    with col3:
        st.write("https://github.com/DataScientest-Studio/sept23_cds_accidents2") 


def onglet_intro():
    
    tab1, tab2, tab3 = st.tabs(["Contexte", "Données source", "Objectifs"]) #, tab3, tab4
    
    with tab1 :
        st.write("**Bilan définitif 2023 de la sécurité routière** publié par \
                  l'Observatoire national interministériel de la sécurité routière (ONISR) \
                  : 3398 décès (-4.3%).")
        
        #st.markdown("- Baisse continue depuis plusieurs années : -4.3% entre 2022 et 2023 (-18% pour les DROMs)")
        st.image('../data/img/courbe_tues_2023.png', width = 600, caption="Source : ONISR, données définitives jusqu'en 2023. \
                 accidents corporels enregistrés par les forces de l'ordre, France métropolitaine")

        st.write("**Des disparités socio-démographiques et territoriales persistantes** :")
        st.markdown("- :man: Hommes : 77,6% des décès")
        st.markdown("- :male-student: 18-24 ans :  91  tués par million d'habitants (vs 48 en moyenne)")
        #st.markdown("- :older_woman: 75 ans ou plus :  77/million d'habitants")
        #st.markdown("- :motorway: Hors agglomération : 59% des décès et 48% des blessés graves")
        st.markdown("- :earth_americas: Outre-mer vs métropole : 91 tués par million d'habitants sur 2019-2023 (vs 46 en métropole)") 

        #st.write("**Selon le mode de transport** :")
        #st.write(":motor_scooter: Les usagers de deux-roues motorisées représentent 22% des personnes tuées pour moins de 2% du trafic motorisé.")
    
        st.write("**et... selon le mode de transport, la météo, le mois de l'année, le type de route etc...**")
        
        # st.write('Au vu de ces constats, plusieurs questions émergent :')
        # col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
        # with col1:
        #     st.write(' ')

        # with col2:
        #     st.markdown(":arrow_right_hook: **:blue[Peut-on prédire la gravité d'un accident selon ses caractéristiques ?]**") #":arrow_right_hook:
        #     st.markdown(":arrow_right_hook: **:blue[Peut-on prédire combien d'accidents vont survenir la semaine prochaine ? le mois prochain ?]**")
        #     st.markdown(":arrow_right_hook: **:blue[Peut-on déterminer les lieux les plus accidentogènes ?]**")

        # with col3:
        #     st.write(' ')

       
        # col1, col2 = st.columns([0.5, 0.5]) 
        # with col1 :
        #     st.image('../data/img/covid_indemnes.png', width = 680)
        # with col2 :
        #     st.image('../data/img/covid_blesseslegers.png', width = 680)
    
        # st.write("#")
        # col1, col2 = st.columns([0.5, 0.5]) 
        # with col1 :
        #     st.image('../data/img/covid_blesseshospitalises.png', width = 680)
        # with col2 :
        #     st.image('../data/img/covid_tues.png', width = 680)
    
    
    with tab2 :
        #network chart https://stackoverflow.com/questions/22920433/python-library-for-drawing-flowcharts-and-illustrated-graphs
        st.write("##### **:blue[Bases de données annuelles des accidents corporels de la circulation routière]**")

        st.write("- Pour chaque **:blue[accident corporel]** en Hexagone et Outre-mer, les forces de l'ordre saisissent des informations \
                 dans un \"**B**ulletin d’**A**nalyse des **A**ccidents **C**orporels\". ")
        
        st.write(":bulb: **:blue[accident corporel]** = accident survenu sur une voie ouverte à la circulation publique, \
                 impliquant au moins un véhicule et **une victime ayant nécessité des soins**.")
        st.write("")

        # \n administré par l’ONISR
        st.graphviz_chart('''
            digraph {
                A1 [label="Bulletin"]
                A2 [label="Bulletin"]    
                A3 [label="..."]  
                A4 [label="Bulletin"]   
                A5 [label="Bulletin"]        
                B [label="Fichier BAAC"]
                C [label="Extraction épurée pour l'open data 
                          'Bases de données annuelles des accidents corporels de la circulation routière'
                           année y, y=2005 à 2022"]
                D1 [label="Caractéristiques_y.csv"]
                D2 [label="Usagers_y.csv"]
                D3 [label="Lieux_y.csv"]
                D4 [label="Véhicules_y.csv"]
                A1 -> B  
                A2 -> B  
                A3 -> B  
                A4 -> B  
                A5 -> B 
                B -> C 
                C -> D1 
                C -> D2
                C -> D3 
                C -> D4 
            }
        ''')

        # st.write("1) **Caractéristiques** : circonstances générales de l'accident")
        # st.write("2) **Lieux** : Description du lieu principal de l'accident") 
        # st.write("3) **Usagers** : caractéristiques des personnes impliquées") 
        # st.write("4) **Vehicules** : caractéristiques des personnes impliqués")    
        
        st.write(":warning: Les **:blue[extractions]** excluent les données permettant la ré-identification des personnes ou \
                 dont la divulgation leur porterait préjudice: vitesse, consommation d'alcool ou de stupéfiants, inattention, malaise... \
                 comportements rapportés dans une grande partie des accidents mortels.")
   
        # st.markdown(":information_source: **Selon l'ONISR, en 2023 les accidents mortels ont impliqué les comportements suivants :**")
        # st.markdown("- vitesse excessive ou inadaptée (28%)\n - consommation d'alcool (22%)\n - inattention (12%) \
        #             \n - consommation de stupéfiants (11%)\n - malaises (11%)")

        st.markdown(":warning: Evolution du codage de la variable **gravité** des blessures entre 2018 et 2019.") 
        st.markdown(":arrow_right_hook:  Analyse des données à partir de 2019 : **:blue[494 182 usagers]**.")
        
        st.caption("https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/")
            
    with tab3 :
        st.write("##### **:blue[Plusieurs axes d'analyse possibles avec ces données]**")

        col1, col2 = st.columns([0.5, 0.5])
        with col1 :
            st.write("Des analyses temporelles...")
            st.image('../data/img/Blesses_legers_365jours.png') # nb_accidents_par_an.png#st.image('../data/img/carte_indemnes_blesses_tues.png')
            st.write()
            st.write("Prédiction de la gravité...")
            st.image('../data/img/repartition.png', width=400) 
        with col2 :
            st.write("Des analyses geographiques...")
            #st.image('../data/img/carte_camembert2.png')
            st.image('../data/img/ui_model_vide.png', width=600)
            #st.html("""<IMG src="../data/img/carte_indemnes_blesses_tues.png" width="500" height="600">""")
            
            st.write(":arrow_right_hook: **:blue[Deux objectifs]** :")
            st.markdown("- Prédire les **:blue[séries temporelles]** quotidiennes du nombre d'accidentés indemnes, blessés légers, \
                        blessés graves et tués.")
            st.markdown("- **:blue[Prédire]** la gravité à partir des caractéristiques de l'accident, du lieu, du véhicule et de l'usager : \
                     optimiser, comparer  et interpréter différents algorithmes de machine learning et de deep learning.")
            
        # with col1 :
        #     st.write()
        #     st.write("Une analyse **géostatstique** rapide montre le potentiel des données pour des prédictions à échelle spatiale fine. Mais techniques hors du cadre de cette formation $\Rightarrow$ concentration sur les méthodes étudiées.")
        # with col2 :
        #     st.image('../data/img/ui_model_vide.png', width=500)
    

