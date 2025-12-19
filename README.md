# Pontia AutoML

Esta es la resolución del ejercicio final del módulo de Machine Learning del Máster de IA, Cloud Computing y DevOps de Pontia, realizado por Daniel Gómez y Daniel Labrador.

## Requisitos previos

Estas son las librerías necesarias (extracto del pyproject.toml generado por UV):

```toml
[project]
name = "pontia-automl"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "ipykernel>=7.1.0",
    "keras>=3.12.0",
    "matplotlib>=3.10.8",
    "numpy>=2.3.5",
    "pandas>=2.3.3",
    "plotly>=6.5.0",
    "scikeras>=0.13.0",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "tensorflow>=2.20.0",
    "tqdm>=4.67.1",
    "xgboost>=3.1.2",
]

[dependency-groups]
dev = [
    "ipykernel>=7.1.0",
    "nbformat>=5.10.4",
]
```

Para poder usar el código:

```bash
git clone https://github.com/dgorozz/pontia-automl.git
cd pontia-automl

uv venv
uv sync
```

Con esto se clona el repositorio y se creará el entorno con las librerías necesarias.

## Uso

### Entrenamiento

Para **entrenar** un modelo, ejecutar el comando:

```bash
uv run python ./scripts/train.py --model <xgb,rf,logreg,tree,rn>
```

Esto entrenará el modelo y lo guardará en la ruta /models/<model_name>/<model_name>.pkl en formato pickle.

### Evaluación

Para **evaluar** los modelos entrenados, ejecutar el comando:

```bash
uv run python ./scripts/eval.py
```

Si quieres que se ploteen la matriz de confusión y la curva ROC, añade el flag `--plot` al final del comando anterior.

Para elegir el mejor modelo, se utiliza la métrica AUC-ROC porque es la más adecuada a modelos probabilísticos, que es el objetivo de los modelos desarrollados: calcular la probabilidad de cancelación.

## Exploratory Data Analysis

En /notebooks hay un notebook, `01_eda.ipynb`, donde se realiza todo el análisis. Se ha intentado hacer de forma que sea lo más autoexplicativo posible. Al final del mismo hay un resumen de todo lo que se ha hecho, tanto para la limpieza como para el tratamiento de las características. Entendemos que con la explicación ofrecida en el propio notebook no es necesario ampliar en este documento.

## Distribución del trabajo

Pese a que los commits sean por parte de uno de la pareja, el trabajo ha sido realizado por ambos de forma equitativa. Todas las decisiones tanto de diseño del código como de la limpieza de datos han sido toamadas de forma conjunta.

## Puntos a mejorar

- **Personalización de los inputs para cada modelo**. Ahora mismo todos los modelos son entrenados con los parámetros por defecto, con la excepción de la semilla aleatoria. Sería un buen punto permitir un JSON, un YAML o unos kwargs para añadir hiperparámetros.
- **Permitir entrenar varios modelos de una tirada**. Definir que el parámetro --model permita introducir una lista de modelos y que los entrene secuencialmente.
- **Mejores visualizaciones**. Dado que los dos tenemos conocimientos básicos de Streamlit, nos hubiera gusta poder hacer una interfaz para manejar la funcionalidad, pero nos hemos quedado cortos de tiempo.
- **Mejor análisis de otros campos**. Nos hubiera gustado entrar más en profundidad en otros campos, sobre todos los categóricos, o incluso haber hecho una especia de estudio geográfico.
