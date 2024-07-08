import streamlit as st 
import zipfile
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest, chi2,f_classif
from sklearn.model_selection import train_test_split

@st.cache_data 
def load_data_raw():
    print("toto")
    dataset_gold = pd.read_csv("../data/data_cleaned.zip", sep = ",", index_col=0, low_memory=False)
    return dataset_gold
    

@st.cache_data
def load_data_gold():
    dataset_raw = pd.read_csv("../data/accidents.zip", sep = ";", low_memory=False)
    return dataset_raw

def explo():
    
    df_raw = load_data_gold()
    df_gold = load_data_raw()
        
    tab1, tab2, tab3 = st.tabs(["Exploration du dataset", "DataVisualisation et traitement", "Analyse interdépendance"])

    with tab1: # Exploration

            st.write("Cette rubrique est décomposée en 2 sections : Présentation des différentes variables (typage, modalités) \
            et Qualité des données sourcées")
            
            with st.expander("**:green[1- PRESENTATION DES VARIABLES]**" , expanded=True):
                liste_variable = ['Num_Acc', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg',
                'int', 'atm', 'col', 'adr', 'lat', 'long', 'catr', 'voie', 'v1', 'v2',
                'circ', 'nbv', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout',
                'surf', 'infra', 'situ', 'vma', 'id_vehicule', 'num_veh', 'place',
                'catu', 'grav', 'sexe', 'an_nais', 'trajet', 'secu1', 'secu2', 'secu3',
                'locp', 'actp', 'etatp', 'id_usager', 'senc', 'catv', 'obs', 'obsm',
                'choc', 'manv', 'motor', 'occutc']
                
                st.write("Le jeu de données est composé de **:blue[55 variables]** reprises dans le diagramme ci-dessous")
                st.image('../data/img/Liste_variables_BDD_gouv.png', width = 800)
                st.write("Nous retrouvons essentiellement des variables catégorielles codées numériquement (un numéro par modalité) et quelques variables continues.")
            
                if st.checkbox("Afficher le tableau récapitulatif des variables") : 
                    df_tableau_variable = pd.read_csv("../data/tableau_variable.csv", index_col = 0)
                    st.dataframe(df_tableau_variable, width = 1000, hide_index=True)

                if st.checkbox("Afficher les différentes modalités d'une variable") : 
                    df_mod_variable = pd.read_csv("../data/modalite_variable.csv")
                    
                    col1, col2 = st.columns([0.3, 0.7])                    
                    
                    with col1 :
                        variable = st.selectbox("Choississez la variable : ", liste_variable)
                        
                        for i in df_mod_variable.Variable :
                            if variable == i :
                                texte = df_mod_variable[df_mod_variable.Variable == i].Description.values[0]
                                st.write(f"**:blue[{texte}]**")
                                break
                    
                    with col2 :
                        for i in df_mod_variable.Variable :
                            if variable == i :
                                df_aff = df_mod_variable[df_mod_variable.Variable == i]
                                df_aff = df_aff.drop(["Variable","Description"], axis=1).reset_index(drop=True)
                                st.dataframe(df_aff, width = 700, hide_index=True)
                                if variable == 'place':
                                    st.image('../data/img/place.png')          
                                break
                            
            with st.expander("**:green[2- QUALITE DES DONNEES SOURCEES]**" , expanded=True):
                
                liste_qual = ["Valeurs manquantes", "Valeurs aberrantes"]
                nan_data = []
                var_nan_data = []
               
                st.write("L'observation de la :blue[qualité] et de la :blue[fiabilité] des données est une étape importante avant les phases de prétraitement et de modélisation.")
                st.write("Cette étape permet de nettoyer le jeu de données pour éviter des :red[erreurs d'interprétation] ou l'apparition de :red[biais] en supprimant les **doublons**, en traitant les **valeurs manquantes** et les **valeurs aberrantes**.")
                
                if st.checkbox("Afficher le nombre de doublons du jeu de données") : # case à cocher pour effectuer une action
                    st.write(df_raw.duplicated().sum())
                
                options1 = st.multiselect("Choisir une ou plusieurs variables à visualiser", liste_variable)            
                options2 = st.multiselect("Sélectionner le type de recherche", liste_qual)            
                
                calcul = st.button(":black-background[Afficher les résultats]", type="secondary")
                
                if calcul:
                    if "Valeurs manquantes" in options2 :
                        for i in options1 :
                            var_nan_data.append(i)
                            nan_data.append(round(df_raw[i].isna().sum()/df_raw.shape[0]*100,2))
                            dict = {'Variable' : var_nan_data, '%Valeurs manquantes' : nan_data}
                            display_df = pd.DataFrame(dict)
                        st.subheader('**:orange[Graphique des valeurs manquantes]**', divider='rainbow')
                        st.write("Graphique des valeurs manquantes")
                        fig = px.bar(display_df, y="%Valeurs manquantes", x="Variable")
                        st.plotly_chart(fig, use_container_width=True) 
                    
                    if "Valeurs aberrantes" in options2 :
                        st.subheader('**:orange[Distribution des valeurs]**', divider='rainbow')
                        st.write("Distribution des valeurs")
                        for i in options1 :
                            fig = px.box(df_raw, x=i)
                            st.plotly_chart(fig, use_container_width=True) 
                            
        
    with tab2:
        st.write("Cette rubrique est décomposée en 3 sections : Description de la variable cible, Visualisation des différentes variables (au regard de leur répartition\
            , de la distribution en fonction de la variable cible) et une présentation des datasets nettoyés utilisés qui seront utilisés pour les différentes modélisations.")
        
        with st.expander("**:green[1- PRESENTATION DE LA VARIABLE CIBLE]**" , expanded=True):
            st.write("Pour notre cas d'usage, dont **la problématique est de définir la gravité d'un accident** et notamment la \
                    gravité de la blessure d'un usager, la variable cible dont les prédictions seront recherchées est la variable **:blue[gravité]** qui est une variable *catégorielle* ou *qualitative*.")
            st.write()
            st.write("Cette variable grav est composée de 4 modalités : Indemne, Blessé Léger, Blessé Grave et Tué.")
            if st.checkbox("Afficher le nombre valeurs manquantes pour la variable") : 
                st.write(df_raw.grav.isna().sum())
            st.write()
            if st.checkbox("Afficher la distribution des différentes classes") :
                st.subheader('**:orange[Distribution des classes de la variable cible]**', divider='rainbow')
                st.write("Distribution de la variable cible")
                x = df_raw.grav.value_counts()
                fig = px.pie(values=x.values, names=['Indemne','Blessé Léger','Blessé Grave', 'Tué'],color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
                st.write()
                st.subheader('**:orange[Distribution de la variable cible]**', divider='rainbow')
                fig = px.box(df_raw, x="grav")
                st.plotly_chart(fig, use_container_width=True)            
            st.write()
            st.write("La variable présente un **:blue[fort déséquilibre de classe]** qu'il faudra prendre en compte lors de la modélisation et ne présente pas de valeurs aberrantes.")
            
        with st.expander("**:green[2- VISUALISATION DES VARIABLES EXPLICATIVES]**" , expanded=True):
            liste_variable = ['jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg',
                            'int', 'atm', 'col', 'adr', 'lat', 'long', 'catr', 'voie', 'v1', 'v2',
                            'circ', 'nbv', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout',
                            'surf', 'infra', 'situ', 'vma', 'place',
                            'catu', 'sexe', 'an_nais', 'trajet', 'secu1', 'secu2', 'secu3',
                            'locp', 'actp', 'etatp', 'senc', 'catv', 'obs', 'obsm',
                            'choc', 'manv', 'motor', 'occutc']
            
            st.write("La *Data Visualisation* consiste à explorer pour chaque variable la **distribution de cette dernière entre ses modalités mais\
                    aussi vis à vis de la variable cible**. L'objectif étant de voir l'influence des différentes modalités sur la variable cible mais aussi\
                    quel **:blue[traitement est à appliquer pour cette variable (suppression, regroupement, recodage...) en vue de la modélisation]**.")
             
            choix = st.selectbox("Sélectionner la variable à visualiser",liste_variable)
            
             
            affichage = st.button(":black-background[Afficher]", type="secondary")
            st.write(":warning: *:red[Attention, certaines variables comme lat, long, ..., comportent un très grand nombre de modalités et donc un temps de run important pour les visualiser]*")
            if affichage :
                df_mod_variable = pd.read_csv("../data/modalite_variable.csv")
                df_traitement_variable = pd.read_csv("../data/tableau_variable_traitement.csv")
                mod_var = df_mod_variable[df_mod_variable.Variable == choix]
                df_aff = mod_var.drop(["Variable","Description"],axis=1).reset_index(drop=True)
                st.dataframe(df_aff, width = 700, hide_index=True)
                if choix == "place":
                    st.image('../data/img/place.png')
                    
                st.subheader('**:orange[Distribution de la variable par modalités]**', divider='rainbow')
                dist_var = df_raw[choix].value_counts().sort_index()
                fig = px.bar(x=dist_var.index,y = dist_var.values, text_auto=True, labels={'x':'Modalités', 'y':"Nombre d'accidents"})
                st.plotly_chart(fig, use_container_width=True)
                st.write()
                st.write("***Commentaires*** :")
                texte = df_traitement_variable[df_traitement_variable.Variable == choix].reset_index(drop=True)['Analyse_1'][0]
                st.write(f"**:blue[{texte}]**")
                st.subheader('**:orange[Distribution de la variable en fonction de la variable cible]**', divider='rainbow')
                st.write("Distribution de la variable en fonction de la variable cible**")
                df_crosstabl = pd.crosstab(df_raw[choix], df_raw.grav, normalize='index')*100
                
                df_crosstabl.columns = ['Indemne', 'Tué', 'Blessé Grave', 'Blessé Léger']
                fig = px.bar(df_crosstabl,y=df_crosstabl.columns,text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                st.write()
                st.write("***Commentaires*** :")
                texte = df_traitement_variable[df_traitement_variable.Variable == choix].reset_index(drop=True)['Analyse_2'][0]
                st.write(f"**:blue[{texte}]**")
                
                st.subheader('**:orange[Traitement appliqué sur la variable]**', divider='rainbow')
                texte = df_traitement_variable[df_traitement_variable.Variable == choix].reset_index(drop=True)['Traitement_appliqué'][0]
                st.write(f"**:blue[{texte}]**")                      
            
        with st.expander("**:green[3- SYNTHESE DU PRETRAITEMENT]**" , expanded=True):
            
            if st.checkbox("Afficher le tableau de synthèse des traitements appliquées sur les variables existantes") : # case à cocher pour effectuer une action
                df_traitement_variable = pd.read_csv("../data/tableau_variable_traitement.csv", index_col = 0)
                df_traitement_variable= df_traitement_variable[["Variable","Traitement_appliqué"]]
                st.dataframe(df_traitement_variable, width=1200)
            
            if st.checkbox("Parcourir les nouvelles variables créées") :
                new_var_df = df_gold
                new_var = ['age_usager','weekend','prox_pt_choc','jour_chome']
                choix = st.selectbox("Sélectionner la variable à visualiser",new_var)
                test = new_var_df[choix].value_counts().sort_index()
                st.subheader('**:orange[Distribution de la variable par modalités]**', divider='rainbow')
                fig = px.bar(x=test.index,y = test.values, text_auto=True, labels={'x':'modalité', 'y':'Nombre'})
                st.plotly_chart(fig, use_container_width=True)
                st.write()
                st.subheader('**:orange[Distribution de la variable en fonction de la variable cible]**', divider='rainbow')
                new_var_df_crosstabl = pd.crosstab(new_var_df[choix], new_var_df.grav, normalize='index')*100
                new_var_df_crosstabl.columns = ['Indemne', 'Tué', 'Blessé Grave', 'Blessé Léger']
                fig = px.bar(new_var_df_crosstabl,y=new_var_df_crosstabl.columns,text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.write("- Cette rubrique décomposée en 2 sections : Analyse statistique des dépendances et synthèse de l'analyse.")
        st.write("- Elle est basée sur le dataset **:green[dataset_final_sans_dummies.csv]** généré à la suite de l'étape de datavisualisation et traitement.")
        st.write("- Pour rappel ce jeu de données contient **:blue[447136 entrées]** et il est composé de 41 variables **dont 33 catégorielles** et quelques variables quantitatives ou continues.")
        
        with st.expander("**:green[1- ANALYSE STATISTIQUE]**" , expanded=True):
            
            liste_variable = [ 'lum', 'dep', 'agg', 'int', 'atm', 'col',
                                'catr', 'circ', 'prof', 'plan', 'surf', 'infra', 'situ',
                                'sexe', 'catv', 'obs', 'obsm', 'manv', 'motor',
                                'weekend', 'eq_ceinture',
                                'eq_casque', 'eq_siege', 'eq_gilet', 'eq_airbag', 'eq_gants',
                                'eq_indetermine', 'eq_autre', 'jour_chome', 'prox_pt_choc']
                        
            st.subheader("**:orange[Indépendance entre les variables catégorielles et la variable cible : Test du Khi2 et p_value]**", divider='rainbow')
            st.write("Ce test permet d'étudier le lien entre 2 variables catégorielles, autrement dit sont-elles indépendantes l'une de l'autre.")
            st.write("**:blue[Question : Existe t'il une influence de la variable X sur la gravité d'un accident pour l'usager ?]**")
            
            col1, col2 = st.columns([0.5, 0.5])
            with col1 :
                st.image('../data/img/Formule_khi2.png', width = 400)
            with col2 :
                st.image('../data/img/Tableau_khi2.png', width = 400)            
            
            choix = st.selectbox("Sélectionner la variable catégorielle pour le test du Khi2 :",liste_variable)                        
            liste_stat = []
            
            st.write("**:red[Tableau de contingence brut]**")
            df_khi = pd.crosstab( df_gold["grav"],df_gold[choix]) 
            st.dataframe(df_khi, width = 1000)
            
            st.write("**:red[Tableau de contingence de situation à l'indépendance]**")
            df_ind = pd.DataFrame(chi2_contingency(df_khi)[3], index=[1,2,3,4])
            df_ind.columns = df_khi.columns
            st.dataframe(df_ind, width = 1000)
            
            st.write("**:red[Tableau des contributions absolues à l'indépendance]**")
            df_ecart = round((df_khi - df_ind)**2/df_ind,1)
            st.dataframe(df_ecart, width = 1000)

            stat, p = chi2_contingency(df_khi)[:2]
            st.write("La valeur du Khi2 pour la variable est de :", stat)
            dll = (df_khi.shape[0]-1) * (df_khi.shape[1]-1)
            st.write("Degré de liberté :", dll)
            st.write("La valeur de la p.value pour la variable est de :", p)
            
            col = ['jour','mois', 'an','grav','grav_rec', 'date', 'heure', 'lat', 'long','age_usager']
            stat_df = df_gold
            for i in stat_df.columns.drop(col) :

                stat, p = chi2_contingency(pd.crosstab(stat_df["grav"], stat_df[i]))[:2]
                a = stat_df["grav"].nunique()
                b =  stat_df[i].nunique()
                V_Cramer = np.sqrt(stat/(pd.crosstab(stat_df["grav"], stat_df[i]).values.sum()*(np.min([a, b]) - 1)))
                liste_stat.append([i, stat, p, V_Cramer])
            
            stat_tot = pd.DataFrame(liste_stat, columns=['Variables', 'stat_Khi2', 'p_value', 'V_cramer']).set_index('Variables').sort_values(by='V_cramer', ascending=False)
            
            st.subheader("**:orange[Intensité des relations entre les variables catégorielles et la variable cible : Test de Cramer]**", divider='rainbow')
            
            col1, col2 = st.columns([0.5, 0.5])
            with col1 :
                st.image('../data/img/Formule_Vcramer.png', width = 400)
            with col2 :
                st.image('../data/img/Tableau_Vcramer.png', width = 400)  
            
            st.write("Le calcul de la valeur de V_Cramer pour notre jeu de données nous montre que :")
            st.write("- Les variables **:blue[eq_ceinture]** et **:blue[eq_casque]** présentent une intensité de relation forte avec la gravité car la valeur est supérieure à 0.3 ")
            st.write("- 3 variables **:orange[catv, obs, et eq_gants]** présentent une intensité de relation moyenne avec la gravité (valeurs comprises entre 0,2 et 0,3)")
            st.write("- **:red[84% des variables]** ont une intensité de relation **faible voire nulle** avec la variable gravité (valeurs inférieures à 0,2)")
            fig = px.bar(data_frame=stat_tot, x=stat_tot.index,y = "V_cramer")
            fig.add_hline(y=0.1, line_color="red")
            fig.add_hline(y=0.2, line_color="orange")
            fig.add_hline(y=0.3, line_color="green")
            st.plotly_chart(fig, use_container_width=True)
                             
        with st.expander("**:green[2- SYNTHESE]**" , expanded=True):
            
            st.write("A travers l'analyse statistique menée, nous avons mis en évidence qu'il y avait peu d'indépendance entre la variable cible et la\
                plupart des variables.")
            st.write("*:blue[Si nous devions laisser la machine choisir pour nous les variables les plus importantes, quelles seraient-elles ?]*")            
            
            choix_df = df_gold
            choix_df.dep = choix_df.dep.replace(["2A","2B"],(201,202))
            choix_df = choix_df.drop(["grav_rec","date"], axis=1)
            X = choix_df.drop('grav',axis=1)
            y = choix_df['grav']
            
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
            
            st.subheader("**:orange[TOP15 des features importances avec SelectKBest de sklearn]**", divider='rainbow')
            sel = SelectKBest(score_func=f_classif, k=15)
            
            sel.fit(X_train,y_train)
            
            mask = sel.get_support(indices=True)
            liste_var = choix_df.columns
            
            liste_var_name = []
            liste_index = []
            
            for i in range(len(mask)):
                liste_index.append(mask[i])
                liste_var_name.append(choix_df.columns[mask[i]])
            
            df_kbest = pd.DataFrame(liste_var_name,index= liste_index, columns=["Variable"])
            st.dataframe(df_kbest, width = 1000) 
