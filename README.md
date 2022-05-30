# Modelo para identificar a los pacientes más probables de ser readmitidos en hospitales
En Chile, entre el 10 y 30% de las personas que son dadas de altas reingresan a los hospitales en un periodo anterior a 30 o 90 días. Según antecedentes bibliográficos, la readmisión puede significar un costo igual a 17.4 mil millones anuales.

Hay que agregar que aquellas personas que son reingresados llegan con un estado de salud deplorable, requiriendo el uso de camas, maquinarias y la atención de especialistas durante un tiempo prolongado, teniendo un impacto negativo sobre el resto de las atenciones.  

El reingreso está asociado a factores tales como la falta de seguimiento o incumplimiento del tratamiento por parte del paciente, entre otros factores.

¿Y si utilizamos un modelo para identificar a los pacientes más probables de ser reingresados, y en base a ello les realizamos seguimiento?

En este repositorio encontraran el desarrollo metodológico para identificar, a través de Machine Learning, aquellos pacientes más probables de ser readmitidos en hospitales. 
En particular:

* Se utilizó el modelo XGboost, una metodología que se basa en árboles de decisiones.

* Se utilizó una base de datos de citas médicas con una tasa de readmisión del 11% (pacientes que vuelven en los primero 30 días, una vez dados de alta), además cuenta con más de 100 mil registros de atención hospitalaria, información del paciente y la marca sobre si es readmitido o no. Los datos fueron descargados desde la página de Kaggle:
https://www.kaggle.com/code/iabhishekofficial/prediction-on-hospital-readmission/notebook

* El modelo fue construido a través de una muestra de construcción, los parámetros y la estabilidad del modelo fue evaluada mediante cross-validation y testeados en una muestra test. El desempeño del modelo fue estimado a través del AUC y la tasa de readmisión en el grupo más riesgoso.

* Dada la poca información disponible, no se pudieron construir variables históricas o relacionadas con el tipo de especialidad que se quiere asistir. quedando pendiente para otro proyecto.


# Cross-validation
Para asegurar la robustez del modelo y su correcta parametrización, se optó por utilizar la metodología de cross-validation, que consiste en utilizar cierto porcentaje de la muestra de desarrollo para entrenar el modelo y el resto para probar el efecto que tiene los parámetros sobre el desempeño del modelo ante datos nuevos. 

[![cross-validation.png](https://i.postimg.cc/4yrXpS3y/cross-validation.png)](https://postimg.cc/QKJL3S1Z)

# Desempeño del modelo Cross-validation
De los resultados obtenidos, se observa que el AUC en cross-validation es bastante similar entre train y test (70 y 67, respectivamente), lo que indicaría que con los parámetros utilizados se está consiguiendo un modelo sin sobreajuste, y bastante bueno, esto a pesar de no contar con variables históricas.

[![AUC-cross-validation.png](https://i.postimg.cc/VvycK6tn/AUC-cross-validation.png)](https://postimg.cc/S2G5smDx)

# Desempeño del modelo y tabla de eficiencia (en test)
Para él modelo se obtuvo un AUC de 71 en train y 67 en test, valores consistentes con lo visto en cross-validation, señal de que se está trabajando con un buen modelo.
A continuación, se muestra la tabla de eficiencia y el número de registros que son readmitidos según tramos de probabilidad. Se puede observar que en el último tramo la tasa de readmisión es del 25% valor que está arriba de la tasa de readmisión global de la base de datos (11%), lo que significa que hay una mayor probabilidad de encontrar pacientes que se van a ausentar en este grupo que si se tomará desde cualquier punto en la base de datos sin un modelo.

[![Tabla-de-desempe-o-en-test.png](https://i.postimg.cc/FKgPMHw0/Tabla-de-desempe-o-en-test.png)](https://postimg.cc/4Kybtsf3)

# Relevancia de las variables y relación con la tasa de readmisión hospitalaria
Ya conociendo el desempeño del modelo, podría ser interesante determinar las variables que contribuyen en mayor medida en la capacidad predictiva del modelo y la forma en que dichas variables se relacionan con la variable respuesta. A continuación, se muestra aquellas variables que contribuyen en mayor medida (ordenadas de mayor a menor)  en la capacidad predictiva del modelo. Entre las más importantes se encuentran: “number_inpatient”, “discharge_disposition_id”, “number_diagnoses”.

[![Gini-porcentual.png](https://i.postimg.cc/CKkQNNFM/Gini-porcentual.png)](https://postimg.cc/DWf6f1JR)

En relación a la variable “number_inpatient” observa que a medida que aumenta el número de hospitalizaciones el año anterior, es mayor la probabilidad de que el paciente sea readmitido en los próximos 30 días.

[![Hospitalizaciones-a-o-anterior.png](https://i.postimg.cc/QNwyWCGJ/Hospitalizaciones-a-o-anterior.png)](https://postimg.cc/YjN35pSv)

En relación a la variable "number_diagnoses" se observa que entre mayor sea el número de diagnósticos que tenga un paciente en el historial del hospital, mayor es la probabilidad de que sea readmitido en los próximos 30 días.

[![Numero-diagnositocos.png](https://i.postimg.cc/wT9s2cpq/Numero-diagnositocos.png)](https://postimg.cc/8sn5Pvc9)

Para mayor detalle, el diccionario de cada variable lo pueden encontrar en el paper que está adjunto a la base de datos en la página de kaggle.

# Estrategia de gestión
Según estos antecedentes se recomienda realizar un seguimiento a aquellos pacientes que se encuentran en el último y penúltimo tramo de mayor probabilidad, pudiendo definirse la estrategia de seguimiento, según la afección y tratamiento recomendado a cada paciente.
Además, se recomienda en un futuro incorporar la fecha de cada registro para poder crear variables históricas, que es bien sabido, incorporar una gran capacidad predictiva sobre los modelos.


# SIGUIENTES ETAPAS
* Incorporar la especialidad médica solicitada
* Construir variables históricas.
