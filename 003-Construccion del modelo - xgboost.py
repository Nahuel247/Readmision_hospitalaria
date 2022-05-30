
###################################################################################
#             PROYECTO: MODELO PARA PREDECIR LA READMISIÓN HOSPITALARIA
#                             CON MACHINE LEARNING
###################################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################


########################################
# IMPORTAMOS LAS LIBRERIAS DE INTERÉS
########################################

import numpy as np
import pandas as pd
np.random.seed(123)
import xgboost as xgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import warnings
warnings.filterwarnings('once')


#############################################
# CREAMOS LAS FUNCIONES QUE VAMOS A UTILIZAR
#############################################


# Definimos función para crear bivariantes
def bivariante(var_exp, var_resp,xlabel, n_tramos):
    if(var_exp.dtypes!="object"):
        # Tramamos la variable explicativa en n_tramos
        bins = list(sorted(set(np.quantile(var_exp.copy(), np.arange(0,1+(1/n_tramos),1/n_tramos),overwrite_input=True))))
        labels = [f'{round(i,3)}-{round(j,3)}' for i, j in zip(bins[:-1], bins[1:])] # creamos etiquetas

        categorias = pd.cut(var_exp, bins=bins, labels=labels, include_lowest=True, right=True)
        df=pd.DataFrame({'var_exp':var_exp,'categorias':categorias,'var_resp':var_resp})
        # agrupamos para conocer la tasa de incumplimiento según tramo
        df_group= df.groupby('categorias').agg(tasa_malo=('var_resp', np.mean), n=('categorias', len)).reset_index()
    else:
        df = pd.DataFrame({'categorias': var_exp, 'var_resp': var_resp})
        df_group = df.groupby('categorias').agg(tasa_malo=('var_resp', np.mean), n=('categorias', len)).reset_index()

    # Graficamos
    matplotlib.rc_file_defaults()
    sns.set_context('poster', font_scale=0.6)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df_group['categorias'], y=df_group['n'], alpha=0.5, ax=ax1, color="blue")
    ax1.set( xlabel=xlabel, ylabel="Número de registros (N)")
    sns.set_context('poster', font_scale=0.6)
    ax2 = ax1.twinx()
    sns.lineplot(x=df_group['categorias'], y=df_group['tasa_malo'], marker='o', sort=False, ax=ax2, color="red")
    ax2.set(ylim=(0, 1), ylabel="Tasa de readmisión hospitalaria")
    fig.show()



def tabla_eficiencia(var_exp,var_resp,n_tramos):
    bins = list(
        sorted(set(np.quantile(var_exp.copy(), np.arange(0, 1 + (1 / n_tramos), 1 / n_tramos), overwrite_input=True))))
    bins[len(bins)-1]=1
    labels = [f'{round(i, 3)}-{round(j, 3)}' for i, j in zip(bins[:-1], bins[1:])]  # creamos etiquetas
    categorias = pd.cut(var_exp, bins=bins, labels=labels, include_lowest=True, right=True)
    df = pd.DataFrame({'var_exp': var_exp, 'rangos_prob': categorias, 'var_resp': var_resp})
    # agrupamos para conocer la tasa de incumplimiento según tramo
    df_group = df.groupby('rangos_prob').agg(n=('rangos_prob', len), n_malos=('var_resp', sum),
                                            tasa_malo=('var_resp', np.mean)).reset_index()
    print(tabulate(df_group, headers=df_group.columns))
    return df_group



##########################################
# CONSTRUIMOS EL MODELO
##########################################

# División de los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(
                                        data_artificial.drop(columns = ["readmitted_1","readmitted_2"]),
                                        data_artificial["readmitted_1"],
                                        random_state = 123)

ids=['encounter_id', 'patient_nbr']

#--------- VALIDACIÓN CRUZADA ---------#

data_dmatrix = xgb.DMatrix(data=X_train.drop(ids,axis=1).copy(),label=y_train)

params = {'objective':'binary:logistic',
          "subsample": 0.8,
          "colsample_bytree": 1,
          'eta': 0.01,
          'max_depth': 3}

# Ajustamos el modelo
xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, metrics = 'auc',early_stopping_rounds=10, num_boost_round=3000, seed=123)



#--------- ENTRENAMOS EL MODELO ---------#

parametros = ({"objective": "binary:logistic",
               "eval_metric": "logloss",
               "subsample": 0.6,
              "colsample_bytree":1,
               "learning_rate": 0.02, #1
               "max_depth": 3,
               "n_estimators": 1333,
               "seed":123
               })

modelo = xgb.XGBClassifier(**parametros)
modelo.fit(X_train.drop(ids,axis=1),y_train)


#################
#   DESEMPEÑO
#################


#------------train ----------#

# Métricas para variable respuesta continua
y_train_pred = pd.DataFrame(modelo.predict_proba(X = X_train.drop(ids,axis=1))).loc[:,1]
roc_auc_score(y_train,y_train_pred)

n_tramos=10
var_exp=y_train_pred
var_resp=y_train.reset_index().copy().readmitted_1

tabla_eficiencia(var_exp,var_resp,10)


#------------test ----------#

# Métricas para variable respuesta continua
y_test_pred = pd.DataFrame(modelo.predict_proba(X = X_test.drop(ids,axis=1))).loc[:,1]
roc_auc_score(y_test,y_test_pred)


n_tramos=10
var_exp=y_test_pred
var_resp=y_test.reset_index().copy().readmitted_1

tabla_eficiencia(var_exp,var_resp,10)



##################################
# Graficamos los resultados
##################################

# Importancia de las variables

feature_important = model.get_booster().get_score(importance_type='total_gain')
keys = list(feature_important.keys())
ganancia_modelo=sum(feature_important.values())
values = np.array(list(feature_important.values()))/ganancia_modelo
data_score = pd.DataFrame({"variable":keys,"valor":values}).sort_values(by = "valor", ascending=False).reset_index()
data_score=data_score.loc[0:10,["variable","valor"]]


sns.set_theme(style="whitegrid", font_scale=1.4)
g = sns.PairGrid(data_score, x_vars="valor", y_vars="variable",
                 height=8, aspect=.25)

g.map(sns.stripplot, size=15, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

g.set(xlim=(0, 0.4), xlabel="Contribución porcentual", ylabel="")
ax.xaxis.grid(False)
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)



# Realizamos bivariantes para entender cómo se relacionan con la tasa de readmisión
#number_inpatient: Number of inpatient visits of the patient in the year preceding the encounter
bivariante(data_artificial.number_inpatient, data_artificial.readmitted_2,"number_inpatient: Número de hospitalizaciones durante el año anterior", 10)

#number_diagnoses: Número de diagnósticos
bivariante(data_artificial.number_diagnoses, data_artificial.readmitted_2,"number_diagnoses: Número de diagnósticos introducidos en el sistema", 10)



