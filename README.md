# Prédiction de la gravité des accidents routiers en France, 2019-2022, par des méthodes d'apprentissage machine

<img src="./data/img/Photo_accueil.png" width="250" height="200">

## Présentation 

Ce dépôt contient les codes de notre projet **ACCIDENTS ROUTIERS EN FRANCE**, développé durant notre [formation Data Scientist](https://datascientest.com/en/data-scientist-course) chez [DataScientest](https://datascientest.com/).

Le principal objectif de ce projet est de  **prédire la gravité des accidents corporels en France**, en utilisant des données historiques sur les accidents corporels recueillies par les forces de l'ordre, et disponibles en [open data](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/).

Notre travail s'est articulé en deux volets :
* l'optimisation et la comparaison d'algorithmes de classification des personnes accidentées en indemne / blessé léger / blessé grave / tué.
* la prédiction des séries temporelles quotidiennes du nombre de personnes dans chacune de ces classes.

Une attention spéciale a été portée à l'identification des facteurs influant le plus sur la gravité des accidents, en utilisant des méthodes d'interprétabilité.

Ce projet a été développé par l'équipe suivante : 

- Matthieu Claudel ([GitHub](https://github.com/) / [LinkedIn](http://www.linkedin.com/in/matthieu-claudel-8a927857))
- Vanessa Ibert ([GitHub](https://github.com/Vanessa-DS) / [LinkedIn](http://www.linkedin.com/in/vanessa-ibert))
- Camille Pelat ([GitHub](https://github.com/) / [LinkedIn](http://www.linkedin.com/in/camille-pelat-08a7b68a))
- Nadège Reboul ([GitHub](https://github.com/) / [LinkedIn](http://www.linkedin.com/in/nadege-reboul))

## Installation
**Cloner le projet**

Vous pouvez cloner le projet grâce à l'instruction suivante qui va créer un dossier sept23_cds_accidents2 dans votre répertoire courant:
```shell
git clone https://github.com/DataScientest-Studio/sept23_cds_accidents2.git
```


## Executer les notebooks de pre-processing, modélisation et datavizualisation
**Création d'un environnement conda**

Après avoir cloné le projet dans le dossier "sept23_cds_accidents2", créez un environnement conda et installez-y les librairies listées dans requirements.txt : 

```shell
conda create --name env_accidents python=3.12
conda activate env_accidents
cd sept23_cds_accidents2 
pip install -r requirements.txt
```

**Configuration du répertoire local des données**

Créez un fichier *global_conf.yml* dans le sous-dossier conf/ de sept23_cds_accidents2, sur le modèle de *global_template.yml*, en indiquant dans la variable *local_data_path* le chemin vers un répertoire local où vous pourrez stocker le fichier de données global.

**Création du fichier de données global**

Exécutez en premier lieu le notebook notebook\Creation_Nettoyage_Preprocessing_Dataset\Creation_dataset.ipynb qui lit les différents jeux de données csv, les concatène et les sauve dans le dossier local.
La base de données est en effet constituées de 4 fichiers (caracteristiques.csv, usagers.csv, vehicules.csv, lieux.csv), par année (de 2019 à 2022), donc 16 fichiers en tout.
Le fichier csv créé, *accidents.csv*, sera appelé dans les autres notebooks.

**Création du fichier nettoyé, pour la modélisation**

Executez ensuite le notebook notebook\Creation_Nettoyage_Preprocessing_Dataset\Nettoyage_dataset.ipynb, qui lit le fichier *accidents.csv*, applique un pre-processing (nettoyage, recodage, création de vairables) et sauve dans le répertoire local choisi les différents fichiers nettoyés, notamment le principal : *data_cleaned_final*, qui sera appelé dans les notebook de modélisation.

## Lancer l'application Streamlit

Lancez les lignes suivantes pour créer un environnement conda dédié à l'application Streamlit, l'activer, y installer les librairies python requises et lancer l'application : 

```shell
conda create --name my-awesome-streamlit python=3.12
conda activate my-awesome-streamlit
cd sept23_cds_accidents2/streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

L'application sera alors disponible à l'adresse suivante [localhost:8501](http://localhost:8501).
