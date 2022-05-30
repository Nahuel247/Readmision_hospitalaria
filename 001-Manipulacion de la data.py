
###################################################################################
#             PROYECTO: MODELO PARA PREDECIR LA READMISIÓN HOSPITALARIA
#                             CON MACHINE LEARNING
###################################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################

########################################
# IMPORTAMOS LAS LIBRERÍAS DE INTERÉS
########################################

import numpy as np
import pandas as pd
import random
from numpy.random import rand
import warnings
import seaborn as sns
from pandas import ExcelWriter
import re
from numpy.random import rand



warnings.filterwarnings('once')

seed=123
np.random.seed(seed) # fijamos la semilla
random.seed(seed)

#############################################
# CREAMOS LAS FUNCIONES QUE VAMOS A UTILIZAR
#############################################


# Definimos una función para procesar una variable categórica
def descriptivo_texto(data,names_column):
    frec = np.unique(data[[names_column]], return_counts=True)
    df = pd.DataFrame(frec).T.copy()
    observaciones=np.nan
    df[observaciones]=np.nan
    df.columns = ["categorías", "frecuencia","observaciones"]
    return df

# Definimos una función para procesar una variable númericas
def descriptivo_num (data, names_column):
    array=data[[names_column]]
    n_zeros = np.sum([array==0]) # N° de ceros
    n_negativos=np.sum([array<0]) # N° de valores negativos
    n_missing=array[np.isnan(array)].sum()[0] # N° de valores missing
    min= array[~np.isnan(array)].min()[0]
    max= array[~np.isnan(array)].max()[0]
    mean= array[~np.isnan(array)].mean()[0]
    p5=np.percentile(array[~np.isnan(array)], 5) # Valor del percentil al 5%
    p10 = np.percentile(array[~np.isnan(array)], 10)
    p25 = np.percentile(array[~np.isnan(array)], 25)
    p50 = np.percentile(array[~np.isnan(array)], 50)
    p75 = np.percentile(array[~np.isnan(array)], 75)
    p90 = np.percentile(array[~np.isnan(array)], 90)
    observaciones=np.nan
    df=pd.DataFrame([{"n_ceros":n_zeros, "n_negativos": n_negativos,
                      "n_missing":n_missing,"min":min,"max":max, "mean":mean,
                      "p5":p5,"p10":p10,"p25":p25,"p50":p50,"p75":p75,"p90":p90,
                      "observaciones": observaciones}]).T
    df.columns=[names_column]
    return df

# construimos una función para gestionar una base de datos con registros categoricos o númericos
def descriptivo(data):
    with ExcelWriter("Descriptivo.xlsx") as writer:
        names_column_num = data.loc[:, (data.dtypes != "object")].columns.copy()
        names_column_text = data.loc[:, (data.dtypes == "object")].columns.copy()
        n = 0
        if (names_column_num.shape[0] >= 1):
            desc_num_list = []
            for names_column in names_column_num:
                desc_num = descriptivo_num(data, names_column)
                desc_num_list.append(desc_num)
            final_desc_num = pd.concat(desc_num_list, axis=1).T
            final_desc_num.to_excel(writer, "var_numericas")
            n += 1

        if (names_column_text.shape[0] >= 1):
            for names_column in names_column_text:
                n = n + 1
                desc_txt = descriptivo_texto(data, names_column)
                desc_txt.to_excel(writer, names_column, index=False)
    print("Se ha creado un descriptivo de los datos")


##############################
# CARGAMOS LOS DATOS
##############################

data=pd.read_csv("diabetic_data.csv",sep=",")

descriptivo(data)

data["race"].replace({"?": "Missing"}, inplace=True)
data["gender"].replace({"Unknown/Invalid": "Missing"}, inplace=True)
data["weight"].replace({"?": "Missing"}, inplace=True)
data["payer_code"].replace({"?": "Missing"}, inplace=True)
data["medical_specialty"].replace({"?": "Missing"}, inplace=True)

medical_speciality=[re.sub("-.*", "", x) for x in data["medical_specialty"]]
data["medical_specialty"]=medical_speciality


data.info()

##############################
# VISUALIZAMOS ALGUNOS CASOS
##############################
#sns.set(font_scale=1.3)

# Comportamiento e interrelación entre la venta de distintos items
#sns.relplot(data=data.query('(item in ["item_1","item_3", "item_8","item_50"])').sample(9000).copy(), x="date",
#            y="sales", hue="item", style="store", palette="tab10",s=80)

# Comportamiento de la demanda para el item_8 en distintas tiendas
#sns.relplot(data=data.query( 'item=="item_8"').sample(8000).copy(), x="date", y="sales",
#             hue="store", palette="tab10", marker="o",edgecolor="none")
