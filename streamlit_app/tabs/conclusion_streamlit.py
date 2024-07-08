import streamlit as st
import pandas as pd
    
def onglet_conclusion():
    with st.expander(":blue[**Objectifs**]" , expanded=True):
        st.write("Les 2 objectifs ont été atteints :")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• **Prédire le nombre d'accidents** dans chaque classe de gravité selon la date via les séries temporelles")
            col1, col2 = st.columns([0.05, 0.95])
            with col2 :
                df = pd.DataFrame(
                    {"Classes" : ["Indemnes", "Blessés légers", "Blessés hospitalisés", "Tués"],
                    "Prévisions à court terme" : ["LSTM", "LSTM", "LSTM", "PROPHET + jours fériés"],
                    "Prévisions à long terme" : ["PROPHET + jours fériés", "MSTL + AutoArima", "PROPHET + jours fériés", "PROPHET + jours fériés"]
                    }
                )
                st.dataframe(df, hide_index = True)
            
            st.write('####')
            st.write("• **Prédire la classification d'un accidenté** dans l'une des 4 classes de gravité via des algorithmes linéaires, de regroupement, d'arbres de décision, des apprentissages d'ensemble et du deep learning")
            col1, col2 = st.columns([0.05, 0.95])
            with col2 :
                df = pd.DataFrame(
                    {"Modèle" : ["Régression Logistique", "LinearSVC", "KNN", "Random Forest", "CatBoost", "XGBoost", "DNN-Keras", "DNN-Pytorch", "DNN with transfomers"],
                     "f1-score Indemnes" :              [0.7562298373001208, 0.7538998504166964, 0.7399676084218103, 0.7648087196866851, 0.7671730998307247, 0.750809973235667,  0.7552568918990423, 0.7526858994681038, 0.7528326585812],
                     "f1-score Blessés légers" :        [0.5945668009255292, 0.6016828415912716, 0.5205965666700795, 0.5401745495495496, 0.5790410868561631, 0.513252589081973,  0.5446711956888751, 0.5600451491767873, 0.5476107429368678],
                     "f1-score Blessés hospitalisés" :  [0.4005342152124141, 0.4211471198192342, 0.3621801686866539, 0.4471862930609917, 0.4447876447876448, 0.393942529925144,  0.4025527736867943, 0.3985575415490749, 0.4173420815643895],
                     "f1-score Tués" :                  [0.2212526662788443, 0.2253106304647952, 0.200449501334457,  0.2589965397923875, 0.2548291796076055, 0.2082658022690437, 0.2196092840957805, 0.2290842205005295, 0.2246202488143515],
                     "Accuracy" :                       [0.6224504401345452, 0.6242127674801403, 0.5655387574361498, 0.6124669004508695, 0.6171008373291348, 0.5723822516437804, 0.5853342159879769, 0.592577959013277,  0.5956866282493793]}
                )
                
                def color_coding(row):
                    return ['background-color:yellow'] * len(row) if row.Modèle == "Random Forest"  else ['background-color:white'] * len(row)
                
                st.dataframe(df.style.apply(color_coding, axis=1), hide_index = True)
                st.write("Nous avons choisi **Random Forest** car c'est le modèle qui obtient le meilleur f1-score pour les tués et les blessés hospitalisés.")
            
    with st.expander(":blue[**Optimisations**]" , expanded=True):        
        st.write("Toutes les optimisations ont amélioré les résultats :")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• Classification binaire :")
            col1, col2 = st.columns([0.05, 0.95])
            with col2 :
                st.image('../data/img/Optimisation_binaire.png')
                st.write("**Augmentation moyenne de 2% de l'accuracy** de chaque jeu de données")
            
            st.write("####")
            st.write("• Création de variables :")
            col1, col2 = st.columns([0.05, 0.95])
            with col2 :
                df = pd.DataFrame(
                    {"Modèle" : ["Random Forest de base", "Ajout de 'nb_usagers_gr", "Ajout de 'catv_percute'"],
                     "f1-score Indemnes" :              [0.7648087196866851, 0.769172787011983,  0.7671141857692834],
                     "f1-score Blessés légers" :        [0.5401745495495496, 0.5350114873638997, 0.5381579505750216],
                     "f1-score Blessés hospitalisés" :  [0.4471862930609917, 0.4524782830863567, 0.4496289519162346],
                     "f1-score Tués" :                  [0.2589965397923875, 0.2636371998896146, 0.2619579794367456],
                     "Accuracy" :                       [0.6124669004508695, 0.6157500178916482, 0.6135672368138553]}
                    )
                
                def color_coding(row):
                    return ['background-color:yellow'] * len(row) if row.Modèle == "Random Forest"  else ['background-color:white'] * len(row)
                
                st.dataframe(df.style.apply(color_coding, axis=1), hide_index = True)
                st.write('**Amélioration du f1-score** pour les tués, les blessés hospitalisés et les indemnes')
                st.write("**Amélioration de l'accuracy** de :")
                st.write("• **0,45%** avec ajout de 'nb_usagers_gr'")
                st.write("• **de 0,14%** avec ajout de 'catv_percute'(mais la modification ne concerne que les piétons soit 7,62% du jeu de données)")
    
    
    with st.expander(":blue[**Interprétation**]" , expanded=True):    
        st.write("Les interprétations ont permis de mettre en évidence que la gravité des accidents est réduite notamment si :")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• **la ceinture de sécurité est utilisée**")
            st.write("• **l'on circule en agglomération**")      

        st.write("Les interprétations détaillées permettent de mieux comprendre les influences de chaque variable sur la gravité de l'accident et de pouvoir faire de la prévention")
            
    with st.expander(":blue[**Perspectives**]" , expanded=True):    
        st.write("Pour améliorer les résultats, on peut envisager de:")
        col1, col2 = st.columns([0.05, 0.95])
        with col2 :
            st.write("• **Combiner l'ajout des variables** 'nb_usagers_gr' **et la classification binaire**")
            st.write("• **Revoir la collecte des données** pour avoir la variables 'catv_percute' pour toutes les personnes accidentées")
            st.write("• **Aggréger de nouvelles sources de données** pour revoir l'utilisation de certaines variables (comme la vitessse au moment de l'accident)")
            st.write("• **Traitement spécifique des données géographiques** par modélisation géostatique")
            
        st.write("Cependant, l'amélioration du jeu de données reste contrainte à la suppression des données influant sur la gravité des accidents (conduite sous dépendance, état de santé...) liée au respect du RGPD")
            

        
            
        
        
        