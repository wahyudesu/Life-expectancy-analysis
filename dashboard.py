import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import requests
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie

# Set seaborn theme
sns.set_theme(context="paper", style="whitegrid")
#Basic
st.set_page_config(page_title="# Life Expectancy Analysis", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
st_lottie(lottie_book, speed=1, height=160, key="initial")

# Load data
df = pd.read_excel('P_Data_Extract_From_World_Development_Indicators.xlsx')

df = df.drop(['Country Code', 'Series Code'], axis=1)
df = df.iloc[:-5]
data = df.pivot(index='Country Name', columns='Series Name', values='2019 [YR2019]')
data.reset_index(inplace=True)

column_names = ['Country', 'CO2 Emission', 'Health Expenditure', 'GDP', 'Immunization DPT', 'Immunization HepB3', 'Immunization Measles', 'Life Expectancy', 'Infant Death', 'Maternal Death', 'Primary Education']
data.columns = column_names

data["Country"] = data["Country"].str.replace(',', '')
data = data.replace(["..", "..."], np.nan)
data.dropna(inplace=True)

population = pd.read_excel("population.xlsx")
data = pd.merge(data, population, on='Country')

cols = data.select_dtypes(exclude=['object'])


row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)

row0_1.title("Analysis Life Expectancy Using Multivariate Linear Regression")

with row0_2:
    add_vertical_space()

row0_2.subheader(
    "A Streamlit web app by [Wahyu Ikbal](https://github.com/wahyudesu), my code repo [here!](https://github.com/wahyudesu/School-project/tree/main/Life%20expectancy%20analisis)"
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    st.markdown(
        "Life expectancy is a basic assessment of the health and well-being of a country's population, which serves as an important indicator to assess the effectiveness of health care systems and economic development. It is important to identify the variables that influence and predict the life expectancy of a country's population in order to develop appropriate strategies to improve the quality and performance of the healthcare system and thereby increase life expectancy. Research conducted to estimate life expectancy focuses on modeling life expectancy using multivariate linear regression or multiple linear regression based on observed health and mortality trends in the population, as well as social, economic, and environmental factors in a country. "
    )
    st.markdown(
        "In this study, we present an innovative approach to analyze life expectancy across countries. Using multiple regression analysis I find the estimated value of the regression coefficient for each unit of the independent variable, assuming the other variables remain constant. In addition, I also test the statistical significance of the regression coefficients, which indicates whether or not there is a significant relationship between the independent variable and the dependent variable. Multiple regression analysis can also be used to predict the value of the life expectancy variable based on known values of the independent variables, using the regression equation that has been formed."
    )
    #st.markdown(
        #"**To begin, please enter the link to your [Goodreads profile](https://www.goodreads.com/) (or just use mine!).** 👇"
    #)

row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
with row2_1:
    need_help = st.expander("What is Multiple Linear Regression? 👉")
    with need_help:
        st.markdown(
            '''Multiple linear analysis is a statistical method used to test the relationship between one dependent variable and two or more independent variables. The purpose of this analysis is to predict the value of the dependent variable based on the values of the independent variables, as well as to determine how much influence the independent variable has on the dependent variable. Multiple linear analysis can also be used to test hypotheses about regression coefficients, which show the slope of the regression line for each independent variable. The general formula of multiple linear analysis is:'''
        )
        st.latex(r''' Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon'''
        )
        st.markdown('''
            Where:
            - $Y$ is the dependent variable we want to predict or explain.
            - $\\beta_0$ is the constant.
            - $\\beta_1X_1, \\beta_2X_2, ..., \\beta_nX_n$ are the independent variables (or predictors) that influence $Y$. Each independent variable $X$ is multiplied by its corresponding coefficient $\\beta$.
            - $\\epsilon$ is the random error unexplained by the model.
        ''')

    st.info(
        """Pastikan untuk menggunakan light theme dengan mengklik tombol titik tiga di pojok kanan atas, lalu klik setting dan pilih Light theme
        """
    )
    ok = st.expander("Dataset here")
    with ok:
        st.dataframe(data)

st.write("")
row9_space1, row9_space2, row9_space3 = st.columns(
    (0.5, 1, 0.5)
)
with row9_space2:
    corr = data.corr()
    mask = np.triu(np.ones_like(corr))
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_alpha(0)
    sns.heatmap(corr, annot=True, cmap="Reds", square=True,)
    st.pyplot(fig)


st.write("")
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

#Plot1
with row3_1:
    st.subheader("Books Read")
    scatter_fig = px.scatter(data, x="Health Expenditure", y="Life Expectancy",
                             size="population", hover_name="Country", log_x=True, size_max=50,
                             title='Scatter Plot: Health Expenditure vs Life Expectancy')

    st.plotly_chart(scatter_fig, theme="streamlit", use_container_width=True)
    st.markdown(
        "It looks like you've read a grand total of  books with authors,** with being your most read author! That's awesome. Here's what your reading habits look like since you've started using Goodreads."
        )

with row3_2:
    st.subheader("Books Read")
    scatter_fig2 = px.scatter(data, x="Immunization DPT", y="Life Expectancy",
                             size="population", hover_name="Country", log_x=True, size_max=50,
                             title='Scatter Plot: Health Expenditure vs Life Expectancy')

    st.plotly_chart(scatter_fig2, theme="streamlit", use_container_width=True)
    st.markdown(
        "It looks like you've read a grand total of  books with authors,** with being your most read author! That's awesome. Here's what your reading habits look like since you've started using Goodreads."
        )

# Download link for notebook
st.markdown("[Download Notebook](https://github.com/maulairfani/product-review-topic-modelling-LDA-curie/archive/refs/heads/main.zip)")
