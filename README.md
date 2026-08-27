# 🎓 Predicción de Selección de Pasantes con Machine Learning

## 🎯 Descripción del Problema

TechNova Solutions recibe un alto volumen de postulantes en cada periodo de reclutamiento de pasantías. Evaluar manualmente el rendimiento académico, las habilidades técnicas, el desempeño en entrevistas y la experiencia extracurricular de cada candidato vuelve el proceso lento, subjetivo e inconsistente, lo que puede llevar a decisiones de contratación poco fundamentadas y pérdida de talento frente a procesos de selección más ágiles.

## 🔧 Solución Propuesta

Se desarrolló un modelo de clasificación de aprendizaje automático que predice si un candidato será seleccionado para una pasantía a partir de sus características académicas, técnicas y personales (CGPA, puntajes de habilidades técnicas y blandas, proyectos, certificaciones, resultados de entrevista, entre otras). Además se construyó una app interactiva en **Streamlit** donde se puede ingresar el perfil de un candidato y obtener la predicción en tiempo real. El objetivo es automatizar y estandarizar el proceso de preselección, reduciendo el tiempo y el sesgo humano en la evaluación inicial de postulantes.

## 📊 Resultados Principales

Modelo: Random Forest (300 árboles), evaluado sobre un conjunto de prueba del 20% (2,000 candidatos).

- **Accuracy:** 73.7%
- **Precision:** 73.7%
- **Recall:** 99.9%
- **F1-score:** 84.9%
- **Variables más importantes:** CGPA (la de mayor peso, con margen), seguida de resume_score, github_score, aptitude_score y linkedin_activity_score — con importancias bastante parejas entre sí.

![Importancia de variables](images/feature_importance.png)

> ⚠️ **Hallazgo importante:** el accuracy (73.7%) está prácticamente igual a la tasa base del dataset (73.7% de candidatos seleccionados históricamente). El recall casi perfecto (99.9%) junto con una precisión igual al accuracy indica que el modelo tiende a predecir "seleccionado" para casi todos los casos, en vez de aprender un patrón real. Esto es consistente con las correlaciones muy bajas (todas menores a 0.07) entre las variables individuales y la variable objetivo `selected` encontradas en el análisis exploratorio. Se documenta como hallazgo del proyecto: el dataset, tal como está, no ofrece señal lineal fuerte por variable individual — quedaría como mejora futura explorar interacciones entre variables, balanceo de clases (SMOTE, undersampling) o un dataset con relaciones más marcadas.

## 🚀 Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/ARTARAUZ/prediccion-seleccion-pasantes.git
cd prediccion-seleccion-pasantes
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el análisis (notebook)

```bash
jupyter notebook src/notebooks/analisis_seleccion_pasantes.ipynb
```

### 4. Ejecutar la app interactiva (Streamlit)

```bash
streamlit run src/app/app.py
```

## 📁 Estructura del Proyecto

```
├── data/
│   └── internship_selection_dataset.csv    # Dataset original (Kaggle)
├── models/                                 # Modelos entrenados guardados (.pkl)
├── src/
│   ├── notebooks/
│   │   └── analisis_seleccion_pasantes.ipynb   # EDA + preprocesamiento + modelado
│   └── app/
│       └── app.py                          # App interactiva en Streamlit
├── images/
│   └── feature_importance.png              # Gráficos exportados del notebook
├── requirements.txt                        # Librerías necesarias
├── setup.py                                # Hace que src/ sea instalable como paquete
├── config.yaml                             # Parámetros del proyecto (rutas, semillas, etc.)
├── .gitignore
└── README.md
```

## 🛠️ Tecnologías Utilizadas

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook
- Streamlit

## 📈 Metodología

1. **Recolección de datos**: dataset público "Internship Selection Prediction Dataset" (Kaggle), 10,000 registros y 21 columnas (20 predictoras + variable objetivo `selected`).
2. **Análisis exploratorio (EDA)**: revisión de tipos de variables, valores nulos, distribución de la variable objetivo (desbalance ~74%/26%), histogramas, boxplots y matriz de correlación.
3. **Preprocesamiento**: imputación de nulos, codificación de variables categóricas (`extracurricular`, `college_tier`, `placement_training`), escalado de variables numéricas y construcción de un pipeline de transformación con `scikit-learn` (`ColumnTransformer`, `Pipeline`).
4. **División de datos**: separación estratificada en conjuntos de entrenamiento y prueba para preservar la proporción de la clase objetivo.
5. **Modelado**: entrenamiento y comparación de modelos de clasificación, evaluados con métricas apropiadas para clases desbalanceadas (precision, recall, F1-score) además de accuracy.
6. **Interpretación**: análisis de importancia de características para identificar los factores con mayor influencia en la decisión de selección.

## 💡 Conclusiones y Aprendizajes

- El modelo Random Forest no logró superar de forma significativa la tasa base del dataset (73.7%), lo que indica que, con las variables disponibles y sin ingeniería de características adicional, no hay un patrón lineal fuerte que separe claramente a los candidatos seleccionados de los no seleccionados.
- CGPA fue la variable con mayor importancia relativa, aunque con un peso moderado (0.10 sobre 1.0); el resto de las variables mostraron importancias muy parejas entre sí, sin ninguna que domine claramente la decisión.
- Este resultado en sí mismo es un hallazgo válido para el proyecto: sugiere que el proceso de selección modelado en el dataset depende de combinaciones no lineales de variables (como plantea la descripción original del dataset) más que de umbrales simples por variable.
- Como mejora futura: probar balanceo de clases (SMOTE), ingeniería de variables de interacción, y modelos que capturen mejor relaciones no lineales (Gradient Boosting, XGBoost) antes de concluir que el dataset no es predictivo.

## 👥 Autores

- Aaron M. Ramirez Mota
- Ariana P. Fallas Calderón
- Jeremy F. Picado Chavarria
- Oscar A. Arauz Cerdas

Proyecto final — curso SC-707 Big Data, Universidad Fidélitas.

## 📚 Fuentes

- Dataset: [Internship Selection Prediction Dataset — Kaggle](https://www.kaggle.com/)
