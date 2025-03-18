import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graficar_barras_linea(df, x_col, y_col_barras, y_col_linea, titulo="Gráfico combinado de barras y línea"):
    """
    Función para graficar un diagrama de barras con una línea superpuesta.
    
    Parámetros:
    - df: DataFrame de Pandas.
    - x_col: Nombre de la columna para el eje x (categorías).
    - y_col_barras: Nombre de la columna para el eje y de las barras (pesos).
    - y_col_linea: Nombre de la columna para el eje y de la línea (variable de respuesta).
    - titulo: Título del gráfico (opcional).
    """

    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Graficar las barras
    ax1.bar(df[x_col], df[y_col_barras], color='skyblue', label=y_col_barras, alpha=0.7)

    # Etiquetas para el eje de las barras
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col_barras, color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Crear un segundo eje Y para la línea
    ax2 = ax1.twinx()  # Comparte el mismo eje X

    # Graficar la línea
    ax2.plot(df[x_col], df[y_col_linea], color='orange', marker='o', label=y_col_linea)

    # Etiquetas para el eje de la línea
    ax2.set_ylabel(y_col_linea, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Título
    plt.title(titulo)

    # Rotar etiquetas del eje X si es necesario
    plt.xticks(rotation=90)

    # Asegurar que no se solapen los gráficos
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

def generar_formato_exposicion_homogenea(df, columna_interes, columna_exposicion, num_beans):
    """
    Genera un formato de beans con exposición homogénea basados en la exposición de otra columna y devuelve la columna
    de interés con el formato aplicado.

    :param df: DataFrame que contiene los datos.
    :param columna_interes: Nombre de la columna de interés que se quiere agrupar en bins.
    :param columna_exposicion: Nombre de la columna que contiene la exposición o peso.
    :param distribucion: Tipo de distribución ('normal', 'uniforme', etc.) para ajustar los beans (parámetro no usado en esta versión).
    :param num_beans: Número de beans o grupos a crear.
    :return: Serie con la columna de interés formateada en beans de exposición constante.
    """
    
    # Ordenar el DataFrame por la columna de interés para crear beans adecuados
    df = df.sort_values(by=columna_interes).reset_index(drop=True)

    # Calcular la exposición total y la exposición por cada bean
    exposicion_total = df[columna_exposicion].sum()
    exposicion_por_bean = exposicion_total / num_beans
    
    # Variable para rastrear la exposición acumulada y asignar beans
    exposicion_acumulada = 0
    limites_beans = []
    current_bean = 1
    limite_actual = df[columna_interes].iloc[0]  # Inicia en el valor mínimo
    
    # Recorrer el DataFrame fila por fila para crear los bins basados en exposición
    for i, fila in df.iterrows():
        exposicion_acumulada += fila[columna_exposicion]
        
        # Si alcanzamos la exposición deseada para este bin, guardar el límite superior
        if exposicion_acumulada >= exposicion_por_bean:
            limite_actual = fila[columna_interes]
            limites_beans.append(limite_actual)
            exposicion_acumulada = 0
            current_bean += 1
            if current_bean > num_beans:
                break  # No más beans después del número establecido
    
    # Si el número de beans es menor al solicitado, añadimos el máximo valor como último límite
    if len(limites_beans) < num_beans:
        limites_beans.append(df[columna_interes].max())
    
    # Usar pd.cut para aplicar los límites de los beans
    df['Columna_formateada'] = pd.cut(df[columna_interes], bins=[-np.inf] + limites_beans, duplicates='drop')
    
    return df['Columna_formateada']