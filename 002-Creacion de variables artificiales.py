
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
warnings.filterwarnings('once')

seed=123
np.random.seed(seed) # fijamos la semilla
random.seed(seed)


####################################################
# CREAMOS LA VARIABLE RESPUESTA
####################################################

data["readmitted_1"] = np.where(data["readmitted"]=="<30", 1, 0)
data["readmitted_2"] = np.where(data["readmitted"]!="NO", 1, 0)


###################################################
# ELIMINAMOS LAS VARIABLES QUE NO VAMOS A UTILIZAR
###################################################

data=data.drop("readmitted",axis=1)

#eliminamos aquellas variables que están representadas principalmente por una categoría
var_eliminar=['diag_1','diag_2','diag_3','acarbose', 'acetohexamide', 'chlorpropamide', 'citoglipton',
'examide', 'glimepiride-pioglitazone', 'glipizide-metformin',
'metformin-pioglitazone', 'metformin-rosiglitazone', 'miglitol','weight',
'tolazamide', 'tolbutamide', 'troglitazone', 'troglitazone','A1Cresult','max_glu_serum']

data=data.drop(var_eliminar,axis=1)


####################################################
# CREAMOS LAS VARIABLES EXPLICATIVAS
####################################################

# Identificamos las variables que son dummys
names_column_text = data.loc[:, (data.dtypes == "object")].columns.copy()

lista=['race', 'gender', 'age', 'payer_code', 'medical_specialty', 'metformin',
       'repaglinide', 'nateglinide', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'insulin', 'glyburide-metformin',
       'change', 'diabetesMed']

data_artificial = pd.get_dummies(data, columns=lista)

regex = re.compile(r"\[|\]|\>|<|\)|\(|>", re.IGNORECASE)
data_artificial.columns = [regex.sub("", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data_artificial.columns.values]

algo=pd.DataFrame(data_artificial.columns)
algo.to_csv("prueba.csv", sep=";")


# Revisamos las dimensiones
data_artificial.shape


# La guardamos para cargar de forma independiente (si se desea)
data_artificial.to_csv("data_artificial.csv", sep=";")