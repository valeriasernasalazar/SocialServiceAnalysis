import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from bokeh.plotting import figure
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

st.set_page_config(page_title='DashBoard OSFS' #metodo para crear la pagina definir el nombre que aparecera en la pestana del navegador del dashboard
                   ,page_icon='moneybag'
                   ,layout='wide' #definir como se vera la pagina en este caso se vera angosta
                   ,initial_sidebar_state='expanded') #esto define que el tab con los filtros estara abierta cuando se corra la aplicacion al principio


def get_data(): #funcion para cargar el csv con el que trabajaremos
  
    df=pd.read_csv('dfdashboard4.csv')
    #Limpieza de datos del archhivo con el que trabajaremods
    df = df.iloc[2:]
    df = df.drop('Unnamed: 0', axis=1)
    #convertir la columna de fecha al formato que reconoce streamlit
    df['Fecha registrada']=pd.to_datetime(df['Fecha registrada'], format='ISO8601').dt.date

    df = df.dropna()
    # Convert question_0 and question_1 columns to int or float
    df['0'] = df['0'].astype(float)
    df['1'] = df['1'].astype(float)
    return df
#mandar a llamar la funcion get_data para guardar el df en la variable df
df=get_data()




#determinar valores necesarios para los filtros 
df.columns = ['Pregunta1', 'Pregunta2'] + list(df.columns[2:]) #renombrar la columna 0,1 por Pregunta1 Pregunta2 
df['Pregunta1'] = df['Pregunta1'] * 100
df['Pregunta1'] = df['Pregunta1'].astype(int) #convertir los puntajes de la pregunta1 una a Integers en el rango de 0-100
df['Pregunta2'] = df['Pregunta2'] * 100
df['Pregunta2'] = df['Pregunta2'].astype(int) #convertir los puntajes de la pregunta2  una a Integers en el rango de 0-100
df = df[df['Pregunta2'] <= 100]

# Assuming df is your DataFrame
df = df.dropna()
# Convert question_0 and question_1 columns to int or float
df.rename(columns={'Fecha registrada': 'Fecha_Registrada'}, inplace=True) 

#valores limites para filtros 
min_date = pd.to_datetime(df['Fecha_Registrada']).min() #esta variable es la fecha de la primera encuesta registrada
max_date = pd.to_datetime(df['Fecha_Registrada']).max() #esta variable es la fecha de la ultima encuesta registrada

min_val_pregunta1=int(df['Pregunta1'].min()) #Esta variable el puntaje de la pregunta1 uno mas bajo
max_val_pregunta1=int(df['Pregunta1'].max()) #Esta variable el puntaje de la pregunta1 uno mas alto


min_val_pregunta2=int(df['Pregunta2'].min()) #Esta variable el puntaje de la pregunta2 uno mas bajo
max_val_pregunta2=int(df['Pregunta2'].max()) #Esta variable el puntaje de la pregunta2 uno mas bajo
df = df[df['OSF'] != '"'] 

df = df[df['OSF'] != '7']  


st.title(':clipboard: OSF DASHBOARD') #titulo del dashboard
st.markdown('##') #enter

#Filtros________________________________________________________________
st.sidebar.header("Opciones a filtrar:") #sidebar lo que nos va a hacer es crear en la parte izquierda un cuadro para agregar los filtros que queremos tener
#Este filtro es para escoger Las OSF que se desean visualizar para ciertas graficas
oSF_filtro = st.sidebar.multiselect(
    "Seleccione la OSF:", #mensaje que aparecera arriba de la caja de seleccion
    options = df['OSF'].unique(), #las opciones que se pueden escoger
    default = ["AMMJE, Asociación Mexicana de Mujeres Jefas de Empresa, A.C.","Dirección de Servicio Social - Aprendizajes para todos"]#Aqui podría por default dejar un filtro especifico pero vamos a dejarlos todos puestos por default
)
start_date = st.sidebar.date_input('Start Date',min_value=min_date,max_value=max_date,value=min_date) #Este filtro generara un calendario donde el usuario puede escoger la primera fecha para un rango donde ver las encuestas
end_date = st.sidebar.date_input('End Date',min_value=min_date,max_value=max_date,value=max_date) #Este filtro generara un calendario donde el usuario puede escoger la segunda fecha para un rango donde ver las encuestas


pregunta1 = st.sidebar.slider( #este filtro sirve para crear un rango de puntajes en las encuestas asi el usuario puede escoger hasta que puntaje desea ver las encuestas
        "Rango de puntaje para Pregunta 1",
        min_value=min_val_pregunta1,
        max_value=max_val_pregunta1,
        value=max_val_pregunta1,
        step=1
)

pregunta2 = st.sidebar.slider(
        "Rango de puntaje para Pregunta 2", #este filtro sirve para crear un rango de puntajes en las encuestas asi el usuario puede escoger hasta que puntaje desea ver las encuestas
        min_value=min_val_pregunta2,
        max_value=max_val_pregunta2,
        value=max_val_pregunta2,
        step=1
)

#__________________________________________________________________________-



#crearemos un nuevo df que esta conectado a los filtros para generar diferentes graficas
df_seleccion1=df[ (df['OSF'].isin(oSF_filtro)) & (df['Pregunta1']<pregunta1)& (df['Pregunta2']<pregunta2) ] #conectar los filtros de las osf y del rango de puntos de las dos preguntas
df_seleccion1=df_seleccion1.query("Fecha_Registrada>=@start_date & Fecha_Registrada<=@end_date") # conectar el filtro de las fechas 





# Esta funcion grafica todas las  predicciones de todas las encuestas, siempre permanecera igual esta conectada al df sin filtro
def satis(df):
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada'])

    alt.data_transformers.disable_max_rows()

    brush = alt.selection_interval()
    points = alt.Chart(df).mark_point().encode(
        x='Fecha_Registrada',
        y=alt.Y('Pregunta1', title='Satisfaccion'),
        color=alt.condition(brush, 'OSF', alt.value('lightgray'))
    ).add_params(
        brush
    )

    st.write(points)

def satis2(df):
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada'])

    alt.data_transformers.disable_max_rows()

    brush = alt.selection_interval()
    points = alt.Chart(df).mark_point().encode(
        x='Fecha_Registrada',
        y=alt.Y('Pregunta2', title='Satisfaccion'),
        color=alt.condition(brush, 'OSF', alt.value('lightgray'))
    ).add_params(
        brush
    )

    st.write(points)
 

st.subheader('Todas las Encuestas') 
col3, col4 = st.columns(2)

with col3: #ingresar datos necesarios para la columna
    st.header('Encuestas Pregunta 1') #titula de la columna 1 
    satis(df)  # mandar a llamar la funcion con graficas para la pregunta 1

with col4: #ingresar datos necesarios para la columna
    st.header('Encuestas Pregunta 2') #titula de la columna 2
    satis2(df) # mandar a llamar la funcion con graficas para la pregunta 2

st.divider()  # 👈 Draws a horizontal rule



#____________________________________________________________________________________________________________

def mejoresOSF(df): # el proposito de esta funcion es desplegar en texto las 5 OSF con mayor puntaje en la prediccion
    #esta grafica toma en cuenta todas las encuestas y no utiliza el df con filtros
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada']) 
    #crear periodos de tiempo escolares
    num_sections = 3 
    min_date = df['Fecha_Registrada'].min()
    max_date = df['Fecha_Registrada'].max()
    date_range = max_date - min_date
    section_length = date_range / num_sections

    df['section'] = ((df['Fecha_Registrada'] - min_date) / section_length).astype(int)
    #sacar el promedio para cada periodo de tiempo de satisfaccion  
    df_means = df.groupby('section')['Pregunta1'].mean().reset_index()
    df_means_sorted = df_means.sort_values('Pregunta1', ascending=False)

    # Initialize an empty DataFrame to store the top 5 OSF for each time period
    top_osf_per_period = pd.DataFrame(columns=['section', 'OSF', 'Mean_Score'])

    # Extract top 5 OSF for each time period
    for section in df['section'].unique():
        section_data = df[df['section'] == section]
        section_means = section_data.groupby('OSF')['Pregunta1'].mean().reset_index()
        section_means_sorted = section_means.sort_values('Pregunta1', ascending=False).head(5)
        section_means_sorted['section'] = section
        top_osf_per_period = pd.concat([top_osf_per_period, section_means_sorted])

    # Reset the index of the resulting DataFrame
    top_osf_per_period.reset_index(drop=True, inplace=True)

    # Display the names for each time period in Streamlit
    for section in df['section'].unique():
        st.markdown(f"**Top 5 OSF Para el Periodo  {section}:**")
        osf_list = top_osf_per_period[top_osf_per_period['section'] == section]['OSF'].tolist()
        for osf in osf_list:
            st.markdown(f"- {osf}")

 
mejoresOSF(df)  


#_____________________________________________________________________________________________________________

def studentsatisfaction1(df): # el proposito de esta funcion es crear una grafica con la satisfaccion promedio de los estudiaantes en cada periodo de tiempoe escolar
    #esta grafica toma en cuenta todas las encuestas y no utiliza el df con filtros
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada']) 
    #crear periodos de tiempo escolares
    num_sections = 2 
    min_date = df['Fecha_Registrada'].min()
    max_date = df['Fecha_Registrada'].max()
    date_range = max_date - min_date
    section_length = date_range / num_sections

    df['section'] = ((df['Fecha_Registrada'] - min_date) / section_length).astype(int)
    #sacar el promedio para cada periodo de tiempo de satisfaccion  
    df_means = df.groupby('section')['Pregunta1'].mean().reset_index()

    #crear grafica de linea para demostrar la satisfaccion
    base = alt.Chart(df_means).mark_point().encode(
        x='section',
        y=alt.Y('Pregunta1', title='Satisfacción Promedio')
       
    )
    st.write(base) # mandar a llamar grafica

  
st.subheader('Promedio Satisfaccion de los Estudiantes')
studentsatisfaction1(df) # mandar a llamar grafica

st.divider()  # 👈 Draws a horizontal rule
st.divider()  # 👈 Draws a horizontal rule 

#crear una seccion para los dfs seleccionadas por filtros
st.title('Busqueda con Filtros')
st.divider()
st.subheader('Encuestas seleccionadas con Filtros:')
st.dataframe(df_seleccion1) #mostrar el df de las encuestas 

st.divider()
#esta funcion graficara las predicciones de la pregunta 1 para las osf seleccionadas y segun los filtros seleccionados, esta grafica es interactiva
def pregunta1(df):
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada'])

    alt.data_transformers.disable_max_rows()

    brush = alt.selection_interval()
    #esta parte crea un scatter plot de todas las predicciones de la pregunta 1 
    points = alt.Chart(df).mark_point().encode(
        x='Fecha_Registrada',
        y='Pregunta1',
        color=alt.condition(brush, 'OSF', alt.value('lightgray'))
    ).add_params(
        brush
    ) #esta funcion crea un bar chart con el numero de encuestas de cada osf en ese rango de tiempo
    bars = alt.Chart(df).mark_bar().encode(
        y='OSF',
        color='OSF',
        x='count(OSF)'
    ).transform_filter(
        brush
    )
    st.write(points & bars) # crea la grafica

#esta funcion graficara las predicciones de la pregunta 2  para las osf seleccionadas y segun los filtros seleccionados, esta grafica es interactiva

def pregunta2(df):
    df['Fecha_Registrada'] = pd.to_datetime(df['Fecha_Registrada'])

    alt.data_transformers.disable_max_rows()

    brush = alt.selection_interval()
        #esta parte crea un scatter plot de todas las predicciones de la pregunta 1 

    points = alt.Chart(df).mark_point().encode(
        x='Fecha_Registrada',
        y='Pregunta2',
        color=alt.condition(brush, 'OSF', alt.value('lightgray'))
    ).add_params(
        brush
    ) #esta funcion crea un bar chart con el numero de encuestas de cada osf en ese rango de tiempo
    bars = alt.Chart(df).mark_bar().encode(
        y='OSF',
        color='OSF',
        x='count(OSF)'
    ).transform_filter(
        brush
    )
    st.write(points & bars) # crea la grafica
#crear dos columnas en el df uno para cada grafica
col1, col2 = st.columns(2)

with col1: #ingresar datos necesarios para la columna
    st.header('Puntaje Pregunta 1') #titula de la columna 1
    pregunta1(df_seleccion1)  # mandar a llamar la funcion con graficas para la pregunta 1

with col2: #ingresar datos necesarios para la columna
    st.header('Puntaje Pregunta 2') #titula de la columna 2
    pregunta2(df_seleccion1) # mandar a llamar la funcion con graficas para la pregunta 2
 
st.divider()  # 👈 Draws a horizontal rule



