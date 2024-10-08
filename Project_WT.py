import streamlit as st
import pandas as pd 
import pickle
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Function to display the cover image and course information
def display_header():
    # Display the cover image
    image = Image.open("Surface world Temperature Analysis.png")
    st.image(image)

    # Caption with course and group information
    st.caption("""**Course** : Data Analyst
    | **Training** : Bootcamp
    | **Cohort** : June 2024
    | **Group** : Annalisa SAFFIOTI,  Gabriela ANGELONI
    """)

# Function for the introduction page
def introduction():
    display_header()  

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
    display_header()  
    st.title("🔎 Data Exploration")
    st.info("To achieve the objectives of this project, we utilized three primary datasets. Below, we present key insights derived from these datasets, covering various aspects of global warming, CO₂ emissions, and the results of our data audit.")
    
    # DF1
    """This project utilizes three main datasets:"""

    with st.expander("**OWID CO2 Data**"):
            """
               - **Volume**: 47,415 rows and 79 columns
                - **Description**: Included comprehensive data on CO₂ emissions and other greenhouse gases, such as methane, for various countries.
                - **Interesting variables**: different sources from CO2, temperature change, population, gdp
                - **Missing values**: more than 50% of the columns of the original dataset presented missing values
                """
            """
                Preview of the OWID CO2 Data File: 
            """
            image = Image.open("df1.png")
            st.image(image)

    # DF2
    with st.expander("**Our World in Data: HADCRUT Surface Temperature Anomaly**"):
                """
                - **Volume**:  29,566 rows and 4 columns
                - **Description**: Contained surface temperature anomaly data from 1880 to the present.
                - **Missing values**: There is no missing values for surface temperature anomaly data
                """

                """
                Preview of the HADCRUT Surface Temperature Anomaly File:
                """
                image = Image.open("df2.png")
                st.image(image)

    # DF4
    with st.expander("**Our World in Data: Continents**"):
                """
                - **Volume**: 285 rows and 4 columns
                - **Description**: This dataset provided information about countries and continents, with the year value being 2015
                - **Missing values**: The dataset included 285 countries with no missing values
                """
                """
                Preview of the Our World in Data: Continents File:
                """

                image = Image.open("df3.png")
                st.image(image)


def visualization():
   display_header()
   st.title("🌏 Visualization")
   st.write("In the Visualization section, we present a series of graphics that offer a visual interpretation of the data, helping to illuminate key trends and patterns. These visuals provide a deeper understanding of the our dataset.")
  
   # Load CSV file
   df = pd.read_csv('df_project.csv')

    # 1) Average Surface temperature Anomaly Over the Years
   st.header("Surface temperature anomaly variation over the years (globally)")
   yearly_temp_change = df.groupby('year')['Surface temperature anomaly'].mean().reset_index()
   
   fig1 = px.line(yearly_temp_change, x='year', y='Surface temperature anomaly',
                  title='Average Surface temperature Anomaly Over the Years',
                  labels={'year': 'Year', 'Surface temperature anomaly': 'Average Temperature Change (°C)'})
   st.plotly_chart(fig1)

   st.write("""
   The chart shows the average surface temperature anomaly over the years. We observe an upward trend in temperature anomalies from 1850 to the present day. In the years before 1900, temperature anomalies were variable and often below zero, indicating cooler temperatures compared to the reference average. Starting from the mid-20th century, there is a gradual and consistent increase in temperature anomalies, with a significant acceleration in recent decades. This suggests an increasingly pronounced global warming trend.
   """)
   # 2) Average Temperature Change Over the Years by Continent
   st.header("How the Surface temperature anomaly changed across the years per continent")
   yearly_continent_temp_change = df.groupby(['year', 'Continent'])['Surface temperature anomaly'].mean().reset_index()


   fig2 = px.line(yearly_continent_temp_change, x='year', y='Surface temperature anomaly',
                  color='Continent', title='Average Temperature Change Over the Years by Continent',
                  labels={'year': 'Year', 'Surface temperature anomaly': 'Average Temperature Change (°C)'})
   st.plotly_chart(fig2)
   st.write("""
   The chart shows the surface temperature anomaly over the years by continent. Here are some key observations:
   - Europe and Asia show higher temperature anomalies with slightly higher medians compared to other continents.
   - Africa has a distribution more centered around zero, indicating less extreme variations in temperature anomalies.
   - North America and South America show a wider distribution with more outliers, indicating greater variability in temperature anomalies.
   - Oceania shows a distribution similar to Africa, with temperature anomalies mostly close to zero, but with some significant outliers.
   """)

   # 3) Surface Temperature Anomaly by Continent (Box Plot)
   st.header("Box Plot of Surface Temperature Anomaly by Continent")
   fig3 = px.box(df, x='Continent', y='Surface temperature anomaly',
                 title='Surface Temperature Anomaly by Continent',
                 labels={'Continent': 'Continent', 'Surface temperature anomaly': 'Surface Temperature Anomaly (°C)'})
   st.plotly_chart(fig3)
   st.write("""
   The box plot chart shows the distribution of surface temperature anomalies for each continent. Here are some observations:
   - Europe and Asia show higher temperature anomalies, with slightly higher medians compared to other continents.
   - Africa has a distribution more centered around zero, indicating less extreme variations in temperature anomalies.
   - North America and South America show a wider distribution with more outliers, indicating greater variability in temperature anomalies.
   - Oceania shows a distribution similar to Africa, with temperature anomalies mostly close to zero, but with some significant outliers.
   This analysis helps in understanding the regional differences in temperature anomaly distributions across different continents.
   """)

   # 4) Average Temperature Change Over the Years by Continent
   st.header("Temperature change over the years by continents")
   yearly_continent_temp_change_ghg = df.groupby(['year', 'Continent'])['temperature_change_from_ghg'].mean().reset_index()


   fig4 = px.line(yearly_continent_temp_change_ghg, x='year', y='temperature_change_from_ghg',
                  color='Continent', title='Average Temperature Change Over the Years by Continent',
                  labels={'year': 'Year', 'temperature_change_from_ghg': 'Average Temperature Change (°C)'})
   st.plotly_chart(fig4)
   st.write("""
   The chart shows the average temperature change over the years, divided by continent. We can observe that:
   - North America exhibits a significant increase in average temperature starting from the mid-19th century, with accelerated growth beginning in the 1950s.
   - Europe and Asia follow a similar pattern, with a steady increase in average temperature over time.
   - South America, Africa, and Oceania show less pronounced but still noticeable increases.
   Overall, all continents show a warming trend, with differences in the magnitude and period of warming acceleration.
   """)

   # 6) Reasons for Temperature Rise (Pie Chart)
   st.header("Pie Chart: Reasons for temperature rise")
   contributions = {
       'Carbon Dioxide (CO2)': df['temperature_change_from_co2'].sum(),
       'Methane (CH4)': df['temperature_change_from_ch4'].sum(),
       'Nitrous Oxide (N2O)': df['temperature_change_from_n2o'].sum()
   }
   labels = list(contributions.keys())
   sizes = list(contributions.values())

   fig6 = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)])
   fig6.update_layout(title_text='Reasons for Temperature Rise')
   st.plotly_chart(fig6)
   st.write("""
   CO2: Significant focus for climate mitigation efforts due to its substantial contribution.
   Methane (CH4): Important to address due to its high warming potential, especially in agriculture and fossil fuels.
   N2O: Requires attention despite smaller contribution, particularly in agriculture.
   """)

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
    
    with st.expander("Metrics performance") :
        st.title("Best performing model: Random Forest Regressor")
        image_mod = Image.open("model_performance_evaluation.png")
        st.image(image_mod)
        st.write("- R²: The model achieves an R² score of 64,60%, indicating it explains a significant portion of the variance. When applied to the test data, the R² score drops to 47,36%, suggesting a decrease in performance and potential overfitting, as previously explained.")
        st.write("- MAE: indicating the average deviation of the model's predictions from the actual surface temperature anomalies is 0,3419")
        st.write("- MSE for the test data is 0.2260, representing the average of the squared differences between predicted and actual values, highlighting the impact of larger errors on the model’s performance.")
        st.write("- RMSE for the test set is 0.4754, indicating the standard deviation of the prediction errors and showing how far the model’s predictions deviate from the actual values on average, providing a sense of accuracy relative to the target variable's units.")
   
    with st.expander("Feature Importances") :
        st.title("Feature Importances")
        image_fi = Image.open("feat_imp.png")
        st.image(image_fi)
        st.write("In terms of **Feature Importances** for the Random Forest, the temperature-related features have very low importance, indicating that the model relies primarily on temporal and demographic data rather than temperature change metrics.")
    
    with st.expander("Scatter Plot") :
        st.title("Scatter Plot Scatter vs. Prediction")
        image_tp = Image.open("target_pred.png")
        st.image(image_tp)
        st.write("The **Scatter Plot of Residuals** suggests that the Random Forest model fits the data well, with consistent prediction errors across the range of values, but there are still challenges with accurately predicting the extremes.")
    
    with st.expander("Residuals") :
        st.title("Scatter Plot of Residuals")
        image_rsp = Image.open("scatter_plot.png")
        st.image(image_rsp)
    
        st.title("Histogram of Residuals")
        image_hr = Image.open("hist_res.png")
        st.image(image_hr)
        
        st.write("- The **Histogram of Residuals** confirms that the Random Forest model performs well, with a good distribution of residuals around zero.")

        st.info("Based on the provided metrics and analysis, the **Random Forest Regressor** was the best-performing model for predicting surface temperature anomaly in this dataset, therefore is the one that we selected to build our Machine Learning model to predict the surface temperature anomaly.")


##### PREDICTION SECTION

# Function to load the trained model
def load_model():
    with open('model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to create an array of features for prediction
def get_features(continent, country, year, population, co2):
    features = np.array([
        continent, country, year, population, co2
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

 # Continent to countries mapping (example mapping, replace with actual data)
    continent_to_countries = {
    'Africa': ['Nigeria', 'Egypt', 'South Africa', 'Angola','Ethiopia'],
    'Asia': ['China', 'India', 'Japan','Indonesia','Bangladesh'],
    'Europe': ['Germany', 'France', 'United Kingdom', 'Russia ','Italy ','Portugal','Spain'],
    'North America': ['United States', 'Canada', 'Mexico'],
    'Oceania': ['Australia', 'New Zealand','Fiji'],
    'South America': ['Brazil', 'Argentina', 'Chile','Peru','Colombia']
 }

 # Continent mapping for model encoding
    continent_mapping = {
    'Africa': 0,
    'Asia': 1,
    'Europe': 2,
    'North America': 3,
    'Oceania': 4,
    'South America': 5
 }

 # Country mapping for model encoding
    country_mapping = {
    'Nigeria': 0,
    'Egypt': 1,
    'South Africa': 2,
    'Angola': 3,
    'Ethiopia': 4,
    'Japan': 5,
    'Indonesia': 6,
    'Bangladesh': 7,
    'China': 8,
    'India': 9,
    'Canada': 10,
    'Mexico': 11,
    'United States': 12,
    'New Zealand': 13,
    'Australia': 14,
    'Brazil': 15,
    'Argentina': 16,
    'Chile': 18,
    'Peru': 19,
    'Colombia': 20,
    'Germany': 21,
    'France': 22,
    'United Kingdom': 23,
    'Russia ': 24,
    'Italy ': 25,
    'Spain': 26,
    'Portugal': 27,
    'Fiji': 28
 }

 # Dropdown for continent
    continent_name = st.selectbox('Continent', list(continent_mapping.keys()))
    continent = continent_mapping[continent_name]

 # Dropdown for country based on selected continent
    country_name = st.selectbox('Country', continent_to_countries[continent_name])
    country_encoded = country_mapping[country_name]

 # Sliders to capture input values for each feature
    year = st.slider('Year', min_value=1851, max_value=2017, value=2020, step=1)  # Treated as an integer
    population = st.slider('Population', min_value=3187.0, max_value=1410275968.0, value=1000000.0)
    co2 = st.slider('CO2', min_value=0.0, max_value=10011.151, value=0.0)

    # Generate prediction when the button is clicked
    if st.button('Predict Surface Temperature'):
        features = get_features(continent, country_encoded, year, population, co2)
        prediction = predict_surface_temperature(features)
        st.write(f"The predicted surface temperature is: {prediction[0]}°C")

        if country_name == 'Angola':
            # Display a custom figure for Angola
            st.subheader("Data for Angola")
            data = {'Country': ['Angola'],
                        'Year': ['2017'],
                        'Population': ['30.208.634'] ,
                        'CO2 Emissions': ['321.700'],
                        'Surface Temperature Anomaly': ['1.22'] }
            
                        
            dfangola = pd.DataFrame(data)
            st.write(dfangola)
            
        elif country_name == 'Germany':
            # Display a custom figure for Germany
            st.subheader("Data for Germany")
            data = {'Country': ['Germany'],
                    'Year': ['2016'],
                    'Population': ['82.331.432'],
                    'CO2 Emissions': ['8.132.500'],
                    'Surface Temperature Anomaly': ['1.45'] 
                    }
                        
            
            dfgermany = pd.DataFrame(data)
            st.write(dfgermany)






def conclusion():
   display_header()  # Call the function to display the header content
   st.title("📌 Conclusion")
   st.info("In conclusion, we aimed to develop a predictive model for surface temperature anomalies using various datasets and machine learning techniques. Among the models tested, the Random Forest emerged as the most effective for predicting surface temperature anomalies.")

   st.write("### Challenges")
   st.write("- **Time Management**: The challenge was balancing the demands of the project with our ongoing learning process. As junior members, we struggled to manage our time effectively, especially when applying newly acquired theoretical knowledge to practical scenarios.")
   st.write("- **Data Merging**: Merging datasets proved difficult, particularly in selecting which variables to include and in handling missing values. This was complicated by overlapping or repetitive information in some of the datasets.")
   st.write("- **Model Performance**: Despite achieving an R² score of 71.77% on the training data, the model's performance dropped significantly on the test data, with an R² score of 52.33%. This suggests potential overfitting and indicates that the model struggled to generalize to new, unseen data.")
   
   st.write("### Learnings and Recommendations")
   st.write("- **Time Management**: A solution could have been establishing a more structured timeline with specific milestones to better distribute tasks and reduce last-minute pressures. Improved planning would have helped in balancing the project workload with ongoing learning.")
   st.write("- **Data Merging**: A more robust strategy for merging datasets could have alleviated initial challenges. This includes careful consideration of variable selection, as well as better handling of missing values to optimize data quality and improve model performance.")
   st.write("- **Model Performance**: Incorporating more relevant features could have enhanced the model’s ability to generalize. Implementing more robust cross-validation techniques might also have provided a better assessment of the model's performance on unseen data, helping to improve its generalization capabilities.")

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
