# Pontia AutoML

Esta es la resolución del ejercicio fina ldel módulo de Machine Learning del Máster de IA, Cloud Computing y DevOps de Pontia, realizado por Daniel Gómez y Daniel Labrador.

## Requisitos previos

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

## Exploatory Data Analysis

En /notebooks hay un notebook, `01_eda.ipynb`, donde se realiza todo el análisis. Se ha intentado hacer de forma que sea lo más autoexplicativo posible. Al final del mismo hay un resumen de todo lo que se ha hecho, tanto para la limpieza como para el tratamiento de las características.