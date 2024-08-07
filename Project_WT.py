import streamlit as st
import pandas as pd 
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import requests
from io import BytesIO
from PIL import Image
from scipy.stats import normaltest
from scipy.stats import norm
from PIL import Image

# Function to display the cover image and course information
def display_header():
    # Display the cover image
    image = Image.open("Surface world Temperature Analysis.png")
    # Visualizza l'immagine in Streamlit
    st.image(image)


    # Caption with course and group information
    st.caption("""**Course** : Data Analyst
    | **Training** : Bootcamp
    | **Cohort** : June 2024
    | **Group** : Annalisa SAFFIOTI,  Gabriela ANGELONI
    """)

# Function for the introduction page
def introduction():
    display_header()  # Call the function to display the header content

    st.title("👋 Introduction")
    st.info("This study aims to leverage machine learning models to predict surface temperature anomalies by identifying key variables that influence temperature changes, ultimately contributing to more informed climate-related decisions.")

    st.write("### Objectives")
    st.write("- Explore different datasets and the quality of the data to identify global patterns")
    st.write("- Data visualization to identify trends and relationships bewteen temperature and factors like GDP, population, CO2")
    st.write("- Develop predictive models for forecasting Surface temperature anomalies")

    st.markdown("""
    <style>
        .right-align {
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)

def data_exploration():
    display_header()  # Call the function to display the header content
    st.title("🔎 Data Exploration")
    st.info("To achieve the objectives of this project, we used 3 main datasets, each providing unique insights into different aspects of global warming and CO₂ emissions and we performed a Data Audit of each of them.")
    
    # DF1
    """This project utilizes three main datasets:"""

    with st.expander("**OWID CO2 Data**"):
            """
               - **Volume**: 47,415 rows and 79 columns
                - **Description**: Included comprehensive data on CO₂ emissions and other greenhouse gases, such as methane, for various countries.
                - **Interesting variables**: different sources from CO2, temperature change, population, gdp
                - **Missing values**: more than 50% of the columns of the original dataset presented missing values
                """
            
            """Preview of the OWID CO2 Data File:"""
            image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\DF1 HEAD.png")
            st.image(image)

    # DF2
    with st.expander("**Our World in Data: HADCRUT Surface Temperature Anomaly**"):
                """
                - **Volume**:  29,566 rows and 4 columns
                - **Description**: Contained surface temperature anomaly data from 1880 to the present.
                - **Missing values**: There is no missing values for surface temperature anomaly data
                """

                """Preview of the HADCRUT Surface Temperature Anomaly File:"""

                image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\DF2 HEAD.png")
                st.image(image)

    # DF4
    with st.expander("**Our World in Data: Continents**"):
                """
                - **Volume**: 285 rows and 4 columns
                - **Description**: This dataset provided information about countries and continents, with the year value being 2015
                - **Missing values**: The dataset included 285 countries with no missing values

                """

                """Preview of the Our World in Data: Continents File:"""

                image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\DF3 HEAD.png")
                st.image(image)

##### PREDICTION SECTION

# Function to load the trained model
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to create an array of features for prediction
def get_features(continent, country, year, population, temperature_change_tot, temperature_change_from_ch4, temperature_change_from_co2, temperature_change_from_ghg, temperature_change_from_n2o, co2):
    features = np.array([
        continent, country, year, population, temperature_change_tot,
        temperature_change_from_ch4, temperature_change_from_co2,
        temperature_change_from_ghg, temperature_change_from_n2o, co2
    ])
    return features.reshape(1, -1)

# Function to make a prediction using the loaded model
def predict_surface_temperature(features):
    model = load_model()
    prediction = model.predict(features)
    return np.round(prediction, 3)


# Streamlit UI for the prediction page
def prediction():
    display_header()  # Call the function to display the header content

    st.title("🔮 Prediction")
    st.write("The Random Forest is an ensemble learning method that combines multiple decision trees to improve the model’s accuracy and robustness. Each tree in the forest is built from a random subset of the data and features, and the final prediction is made by averaging the predictions of all individual trees.")
    st.info("Our model was fine-tuned using GridSearchCV to optimise its hyperparameters and improve performance.")

    # Sliders to capture input values for each feature
    continent = st.slider('Continent', min_value=0.0, max_value=5.0, value=0.0, step=1.0)
    country = st.slider('Country', min_value=0.0, max_value=190.0, value=0.0, step=1.0)
    year = st.slider('Year', min_value=1851, max_value=2017, value=2020, step=1)  # Treated as an integer
    population = st.slider('Population', min_value=3187.0, max_value=1410275968.0, value=1000000.0)
    temperature_change_tot = st.slider('Total Temperature Change', min_value=-0.002, max_value=0.541, value=0.0)
    temperature_change_from_ch4 = st.slider('Temperature Change from CH4', min_value=-0.001, max_value=0.056, value=0.0)
    temperature_change_from_co2 = st.slider('Temperature Change from CO2', min_value=0.0, max_value=0.225, value=0.0)
    temperature_change_from_ghg = st.slider('Temperature Change from GHG', min_value=-0.001, max_value=0.271, value=0.0)
    temperature_change_from_n2o = st.slider('Temperature Change from N2O', min_value=0.0, max_value=0.01, value=0.0)
    co2 = st.slider('CO2', min_value=0.0, max_value=10011.151, value=0.0)

    # Generate prediction when the button is clicked
    if st.button('Predict Surface Temperature'):
        features = get_features(continent, country, year, population, temperature_change_tot, temperature_change_from_ch4, temperature_change_from_co2, temperature_change_from_ghg, temperature_change_from_n2o, co2)
        prediction = predict_surface_temperature(features)
        st.write(f"The predicted surface temperature is: {prediction[0]}°C")



####### PREDICTION

# Placeholder functions for other pages

def visualization():
    display_header()  # Call the function to display the header content
    st.title("🌏 Visualization")
    st.write("Visualization content goes here.")

    st.title("Surface temperature anomaly variation over the years (globally)")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\Surface temperature anomaly variation over the years (globally).png")
    st.image(image)
    """
    The chart shows the average surface temperature anomaly over the years. We observe an upward trend in temperature anomalies from 1850 to the present day. In the years before 1900, temperature anomalies were variable and often below zero, indicating cooler temperatures compared to the reference average. Starting from the mid-20th century, there is a gradual and consistent increase in temperature anomalies, with a significant acceleration in recent decades. This suggests an increasingly pronounced global warming trend.
      """
    st.title("How the Surface temperature anomaly changed across the years per continent")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\How the Surface temperature anomaly changed across the years per continent.png")
    st.image(image)
    

    st.title("Box Plot of Surface Temperature Anomaly by Continent")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\Box Plot of Surface Temperature Anomaly by Continent.png")
    st.image(image)
    """
    The box plot chart shows the distribution of surface temperature anomalies for each continent. Here are some observations:

    - Europe and Asia show higher temperature anomalies, with slightly higher medians compared to other continents.
    - Africa has a distribution more centered around zero, indicating less extreme variations in temperature anomalies.
    - North America and South America show a wider distribution with more outliers, indicating greater variability in temperature anomalies.
    - Oceania shows a distribution similar to Africa, with temperature anomalies mostly close to zero, but with some significant outliers.

    This analysis helps in understanding the regional differences in temperature anomaly distributions across different continents.
        """
    st.title("Temperature change over the years by continents")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\temperature change over the years by continents.png")
    st.image(image)

    """ The chart shows the average temperature change over the years, divided by continent. We can observe that:

    - North America exhibits a significant increase in average temperature starting from the mid-19th century, with accelerated growth beginning in the 1950s.
    - Europe and Asia follow a similar pattern, with a steady increase in average temperature over time.
    - South America, Africa, and Oceania show less pronounced but still noticeable increases.

    Overall, all continents show a warming trend, with differences in the magnitude and period of warming acceleration.
        """
    st.title("Pie Chart: Reasons for temperature rise")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\reasons for temperature rise.png")
    st.image(image)

    """
    CO2: Significant focus for climate mitigation efforts due to its substantial contribution.
    Methane (CH4): Important to address due to its high warming potential, especially in agriculture and fossil fuels.
    N2O: Requires attention despite smaller contribution, particularly in agriculture.
        """

    st.title("Line Plot of tot CO2 emission by Top 5 Countries")
    image = Image.open("C:\\Users\\gabri\\Desktop\\Project_WT\\Project_WT\\images\\Line Plot of tot CO2 emission by Top 5 Countries.png")
    st.image(image)

    """
    The graph shows CO2 emissions over the years for the top 5 countries:
    - China: Rapid increase starting in the late 1990s, becoming the highest emitter due to rapid industrialization.
    - United States: Steady rise from the late 19th century, peaking around 2000, then slightly declining.
    - Russia: Increase from the early 20th century, with significant rises post-1990.
    - Germany: Steady increase until the 1970s, followed by a decline due to environmental regulations.
    - United Kingdom: Early rise peaking mid-20th century, then a gradual decline.
    """


def modeling():
    display_header()  # Call the function to display the header content
    st.title("🧩 Modelling")
    st.info("Our goal: to identify the best performing machine learning model for predicting surface temperature anomalies using various regression techniques.")
    st.write("### Methodology")
    st.write("- First check of the missing values and modalities of our merged dataset")
    st.write("- Encoded the categorical variables with get_dummies")
    st.write("- Set **Surface Temperature Anomaly** as target variable to evaluate the performace of our models")
    st.write("- Split the dataset into training and testing sets")
    st.write("- Evaluated test and training sets of all our models are based on their **R² Scores, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)**")

    st.write("### Results")
    st.write("In the table below you can find the results from the different Model presented: ")
    
    if st.button("Metrics performance") :
        image_mod = Image.open("model_performance_evaluation.png")
        st.image(image_mod)

    st.write("### Best performing model: Random Forest Regressor")
    

    st.write("- In terms of **Feature Importances** for the Random Forest, as showed in the chart, the temperature-related features have very low importance, indicating that the model relies primarily on temporal and demographic data rather than temperature change metrics.")
    if st.button("Feature Importances") :
        image_fi = Image.open("feat_imp.png")
        st.image(image_fi)
    
    st.write("- The **Scatter plot Target Vs. Residuals** shows that the Random Forest model generally performs well, with many predictions closely aligning with actual values, but there are still challenges with accurately predicting the extremes.")
    if st.button("Scatter Plot Scatter vs. Prediction") :
        image_tp = Image.open("target_pred.png")
        st.image(image_tp)
   
    st.write("- As the **Scatter Plot of Residuals** suggests that the Random Forest model fits the data well, with consistent prediction errors across the range of values.")
    if st.button("Scatter Plot of Residuals") :
        image_rsp = Image.open("scatter_plot.png")
        st.image(image_rsp)

    st.write("- The **Histogram of Residuals** confirms that the Random Forest model performs well, with a good distribution of residuals around zero.")
    if st.button("Histogram of Residuals") :
        image_hr = Image.open("hist_res.png")
        st.image(image_hr)
    
    st.info("Based on the provided metrics and analysis, the **Random Forest Regressor** was the best-performing model for predicting surface temperature anomaly in this dataset, therefore is the one that we selected to build our Machine Learning model to predict the surface temperature anomaly.")
 


def conclusion():
    display_header()  # Call the function to display the header content
    st.title("📌 Conclusion")
    st.info("In conclusion, we aimed to develop a predictive model for surface temperature anomalies using various datasets and machine learning techniques. Among the models tested, the Random Forest emerged as the most effective for predicting surface temperature anomalies.")

    st.write("### Challenges")
    st.write("- Numerous datasets to analyze, with some containing overlapping or repetitive information")
    st.write("- Data merging presented initial difficulties, especially in regards to the selections of the variables to include and the best strategy to handle missing values.")
    st.write("### Recommandations")
    st.write("- **Data Selection**: Careful consideration should be given to the selection of datasets, ensuring that redundant or overlapping data is minimized.")
    st.write("- **Inclusion of Additional Variables**: Future efforts could focus on identifying and incorporating additional relevant variables to enhance the predictive power of the models.")
    st.write("- **Merging Strategy**: A more robust strategy for merging datasets could help in overcoming initial challenges and optimizing the data for better model performance.")


    st.markdown("""
    <style>
        .right-align {
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)

# Main function for navigation
def main():
    # Dictionary mapping page names to functions
    pages = {
        "👋 Introduction": introduction,
        "🔎 Data Exploration": data_exploration,
        "🌏 Visualization": visualization,
        "🧩 Modelling": modeling,
        "🔮 Prediction": prediction,
        "📌 Conclusion": conclusion
    }

    # Sidebar navigation menu
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page
    if selected_page in pages:
        pages[selected_page]()

# Entry point for the application
if __name__ == "__main__":
    main()
