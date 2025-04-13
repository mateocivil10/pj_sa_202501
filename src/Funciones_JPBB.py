import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from scipy.stats import kstest
import statsmodels.formula.api as smf  # Para fit_glm_poisson
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor  # Para mejorar la velocidad
from IPython.display import display  # Para mostrar mejor las tablas en Colab

#Carga la base de datos parquet
#Variables de entrada
#directory_path (str): Ruta del directorio donde están los archivos Parquet.
#file_pattern (str, optional): Patrón de nombres de archivo (default: 'part-*.parquet').
#columns (list, optional): Lista de columnas a cargar (default: None, carga todas).

def load_and_process_data(directory_path, file_pattern="part-*.parquet", columns=None):
    file_paths = sorted(glob.glob(os.path.join(directory_path, file_pattern)))

    if not file_paths:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {directory_path} con el patrón {file_pattern}")

    df_list = []
    for file_path in file_paths:
        try:
            print(f"📂 Cargando archivo: {file_path}")
            df = pd.read_parquet(file_path, engine="pyarrow", columns=columns)

            # Aplicar filtros de calidad
            df = df[(df["exp_corr_ACAGBC"] > 0) & (df["stro_Corr_AGUAACAGBC"] > 0)]

            # Crear variables calculadas
            df["frecuencia"] = df["stro_Corr_AGUAACAGBC"] / df["exp_corr_ACAGBC"]
            df["severidad"] = df["CUPD_CAP_Corr_aguaacagbc"] / df["stro_Corr_AGUAACAGBC"]

            # Convertir a tipo string para categorización
            df["Etiq_AGUAACAGBC"] = df["Etiq_AGUAACAGBC"].astype(str)

            df_list.append(df)

        except Exception as e:
            print(f"❌ Error al cargar {file_path}: {e}")
            continue

    if not df_list:
        raise ValueError("❌ No se pudo cargar ningún archivo Parquet correctamente.")

    # Concatenar todos los DataFrames procesados
    df_final = pd.concat(df_list, ignore_index=True)

    print(f"✅ Datos cargados y procesados correctamente. Total de filas: {df_final.shape[0]}")
    return df_final



def transform_variables(df, log_transform_cols=None, standardize_cols=None, 
                        categorical_cols=None, numeric_no_transform_cols=None):

    # Inicializar listas vacías si no se proporcionan
    log_transform_cols = log_transform_cols or []
    standardize_cols = standardize_cols or []
    categorical_cols = categorical_cols or []
    numeric_no_transform_cols = numeric_no_transform_cols or []

    # Verificar que las columnas especificadas existan en el DataFrame
    all_specified_cols = set(log_transform_cols + standardize_cols + categorical_cols + numeric_no_transform_cols)
    missing_cols = [col for col in all_specified_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Las siguientes columnas especificadas no están en el DataFrame: {missing_cols}")

    # Transformaciones logarítmicas (log1p para manejar valores cero)
    for col in log_transform_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    # Estandarización
    for col in standardize_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Convertir a categóricas
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Asegurarse de que las columnas numéricas sin transformación sean numéricas
    for col in numeric_no_transform_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def split_data(df, test_size=0.2, random_state=42):

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

 #   Ajusta un modelo lineal generalizado (GLM) con distribución de Poisson.

#   Parámetros:
#   train : DataFrame
#       Conjunto de datos de entrenamiento.
#   formula : str
#       Fórmula de `statsmodels` que define la relación entre la variable dependiente y las independientes.
#   offset_col : str, opcional
#      Nombre de la columna que servirá como offset en el modelo (por defecto, 'exp_corr_ACAGBC').

#    Retorna:
#    model : statsmodels.genmod.generalized_linear_model.GLMResults
#       Modelo ajustado.

def fit_glm_poisson(train, formula, offset_col='exp_corr_ACAGBC'):
    model = smf.glm(formula=formula,
                    data=train,
                    family=sm.families.Poisson(),  # Usamos Poisson
                    offset=np.log(train[offset_col])).fit()
    return model
    
def evaluate_model(model, test, response_col='stro_Corr_AGUAACAGBC'):

    # Predicciones
    test['pred'] = model.predict(test)

    # MAE
    mae = np.mean(np.abs(test[response_col] - test['pred']))

    # KS Test
    mu = model.fittedvalues
    ks_stat, ks_pvalue = kstest(train[response_col], 'poisson', args=(mu,))

    # Sobredispersión
    dispersion = model.pearson_chi2 / model.df_resid

    return {
        'MAE': mae,
        'KS_stat': ks_stat,
        'KS_pvalue': ks_pvalue,
        'Dispersion': dispersion
    }

def calculate_expected_frequency(df, model, group_cols):

    df['frecuencia_predicha'] = model.predict(df)
    e_f_perfil = df.groupby(group_cols)['frecuencia_predicha'].mean().reset_index()
    return e_f_perfil



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def generar_resumen_exploratorio(df, factores):

    numerico_sum = []
    categoria_sum = []

    for col in factores:
        if df[col].dtype in ['int64', 'float64']:  # Variable numérica
            q1, q3 = df[col].quantile([0.1, 0.9])
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col].count()
            total_count = df[col].count()
            outlier_pct = (outliers / total_count * 100) if total_count > 0 else 0

            numerico_sum.append({
                "Variable": col,
                "Tipo": "Numérico",
                "Nulos": df[col].isnull().sum(),
                "Cantidad": total_count,
                "Promedio": round(df[col].mean(), 2),
                "Moda": df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
                "Min": round(df[col].min(), 2),
                "Max": round(df[col].max(), 2),
                "Outliers (IQR)": outliers,
                "% Outliers": f"{round(outlier_pct, 2)}%" + (" ⚠️" if outlier_pct > 10 else "")
            })
        
        else:  # Variable categórica
            categoria_sum.append({
                "Variable": col,
                "Tipo": "Categórico",
                "Nulos": df[col].isnull().sum(),
                "Cantidad": df[col].count(),
                "Moda": df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
                "#Categorías": df[col].nunique()
            })

    # Convertir listas a DataFrame y ordenar
    summary_df = pd.DataFrame(numerico_sum + categoria_sum)
    summary_df["Tipo_Orden"] = summary_df["Tipo"].map({"Numérico": 0, "Categórico": 1})
    summary_df = summary_df.sort_values(by=["Tipo_Orden", "Variable"]).drop(columns=["Tipo_Orden"])
    
    # Aplicar formato visual con colores
    def resaltar_nulos(val):
        """Resalta en rojo los valores nulos"""
        return 'background-color: red' if pd.isnull(val) else ''

    styled_summary = summary_df.style.set_properties(**{'text-align': 'center'}) \
                                     .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]) \
                                     .background_gradient(cmap="coolwarm", subset=["Promedio", "Min", "Max"]) \
                                     .applymap(resaltar_nulos)  # ✅ Ahora sin errores

    return styled_summary  # Devolvemos el objeto estilo para visualizar en Colab



#Genera un grid de boxplots para un conjunto de variables numéricas en un DataFrame 
#    Parámetros:
#    df (pd.DataFrame): DataFrame con los datos.
#    factores (list): Lista de columnas numéricas a graficar.
#    figsize (tuple): Tamaño de la figura (ancho, alto). Default (15, 10).
#    columnas (int): Número de columnas en la grilla de subplots. Default 3.
#    Retorna:
#    Muestra los boxplots organizados en un layout adecuado.
def generar_boxplots(df, factores, figsize=(15, 10), columnas=3):
    """
    """
    filas = math.ceil(len(factores) / columnas)  # Determinar filas según cantidad de variables
    fig, axes = plt.subplots(filas, columnas, figsize=figsize)
    axes = axes.flatten()  # Aplanar matriz de ejes para acceso secuencial

    def plot_boxplot(i, col):
        """Función para graficar un boxplot en un subplot específico."""
        sns.boxplot(y=df[col], ax=axes[i], color=plt.cm.Set3(i % 12))  # Ciclo de colores Set3
        axes[i].set_title(col)
    
    # Paralelizar la creación de gráficos para mayor rapidez
    with ThreadPoolExecutor() as executor:
        for i, col in enumerate(factores):
            executor.submit(plot_boxplot, i, col)

    # Ajustar diseño y eliminar ejes vacíos
    for j in range(len(factores), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#    Ajusta los valores atípicos en variables numéricas mediante winsorización.

#    Parámetros:
#    df (pd.DataFrame): DataFrame con los datos.
#    factores (list): Lista de columnas a procesar.
#    percentil_bajo (float): Percentil inferior para winsorización (default 0.5%).
#    percentil_alto (float): Percentil superior para winsorización (default 99.5%).

#    Retorna:
#    pd.DataFrame: DataFrame con los outliers ajustados.

def ajustar_outliers_winsorizacion(df, factores, percentil_bajo=0.005, percentil_alto=0.995):

    df_ajustado = df.copy()
    
    for col in factores:
        if df_ajustado[col].dtype in ['int64', 'float64']:  # Solo para variables numéricas
            q_low = df_ajustado[col].quantile(percentil_bajo)
            q_high = df_ajustado[col].quantile(percentil_alto)
            df_ajustado[col] = np.clip(df_ajustado[col], q_low, q_high)

    return df_ajustado


#    Genera un resumen de los valores nulos en las columnas especificadas del DataFrame.

#    Parámetros:
#    df (pd.DataFrame): DataFrame con los datos.
#    factores (list): Lista de columnas a analizar.

#    Retorna:
#    pd.DataFrame: Tabla ordenada con el número y porcentaje de valores nulos.
def resumen_valores_nulos(df, factores):

    missing_summary = df[factores].isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100
    
    resumen = pd.DataFrame({
        "Valores Nulos": missing_summary,
        "Porcentaje (%)": missing_percentage
    })
    
    resumen = resumen[resumen["Valores Nulos"] > 0].sort_values(by="Valores Nulos", ascending=False)

    # Mostrar en formato visual atractivo en Colab
    if not resumen.empty:
        from IPython.display import display
        display(resumen.style.background_gradient(cmap="Reds"))

    return resumen

#   Genera una matriz de correlación entre los factores y las variables objetivo con un diseño atractivo.

#    Parámetros:
#    df (pd.DataFrame): DataFrame con los datos.
#    factores (list): Lista de nombres de columnas numéricas a analizar.
#    objetivo (list): Lista de variables objetivo a incluir en la correlación.
#   cmap (str): Paleta de colores para el heatmap. Default "coolwarm".
    
#    Retorna:
#    - Lista de los 5 factores más correlacionados con cada variable objetivo.
#    - Gráfico de la matriz de correlación.
def analizar_correlacion(df, factores, objetivo=["frecuencia", "severidad"], cmap="coolwarm"):

    # Filtrar solo variables numéricas
    num_vars = [col for col in factores if df[col].dtype != 'object']
    
    # Matriz de correlación
    cor_matrix = df[num_vars + objetivo].corr()

    # Selección de los 5 factores más correlacionados con cada variable objetivo
    top_frecuencia = cor_matrix["frecuencia"].drop("frecuencia").abs().sort_values(ascending=False).head(5).index.tolist()
    top_severidad = cor_matrix["severidad"].drop("severidad").abs().sort_values(ascending=False).head(5).index.tolist()

    print("\n🔍 **Top factores correlacionados con FRECUENCIA:**", top_frecuencia)
    print("🔍 **Top factores correlacionados con SEVERIDAD:**", top_severidad)

    # 🎨 Visualización del heatmap mejorado
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        cor_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
        linewidths=1, linecolor="white", cbar_kws={"shrink": 0.8}
    )
    plt.title("🔍 Matriz de Correlación entre Factores y Variables Objetivo", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()

    return top_frecuencia, top_severidad
