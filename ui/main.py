import time
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split
import pickle
import os
import plotly.express as px


@st.cache_data
def load_data():
    raw_df = pd.read_csv(os.path.join("..", "Datasets", "listings.csv"))
    df = pd.read_csv(os.path.join("..", "Datasets", "listings_new.csv"))
    return raw_df, df


@st.cache_resource
def load_model(model_name):
    return pickle.load(open(os.path.join("..", "Models", model_name), "rb"))


# @st.cache_data
def calculate_metrics(_model, X_train, y_train, x_test, y_test):
    model_train_cv = cross_validate(_model, X_train, y_train, scoring=[
        "neg_mean_squared_error", "neg_mean_absolute_error", "r2"], cv=5)
    training_rmse = sum(
        np.sqrt(-model_train_cv['test_neg_mean_squared_error'])) / len(model_train_cv['test_neg_mean_squared_error'])
    training_mae = sum(
        -model_train_cv['test_neg_mean_absolute_error']) / len(model_train_cv['test_neg_mean_absolute_error'])
    training_r2 = sum(model_train_cv['test_r2']) / \
        len(model_train_cv['test_r2'])

    model_test_cv = cross_validate(_model, x_test, y_test, scoring=[
        "neg_mean_squared_error", "neg_mean_absolute_error", "r2"], cv=5)
    testing_rmse = sum(
        np.sqrt(-model_test_cv['test_neg_mean_squared_error'])) / len(model_test_cv['test_neg_mean_squared_error'])
    testing_mae = sum(
        -model_test_cv['test_neg_mean_absolute_error']) / len(model_test_cv['test_neg_mean_absolute_error'])
    testing_r2 = sum(model_test_cv['test_r2']) / len(model_test_cv['test_r2'])

    return [(training_rmse, training_mae, training_r2), (testing_rmse, testing_mae, testing_r2)]


def display_metrics(parent, model, X_train, y_train, X_test, y_test):
    with parent.container(border=True):
        cols = st.columns(3)
        training_metrics, testing_metrics = calculate_metrics(
            model, X_train, y_train, X_test, y_test)
        cols[0].metric(label="Training RMSE", help="Generally, the lower the RMSE, the better the model. It's a good measure of how well the model is performing.",
                       value=training_metrics[0])
        cols[1].metric(label="Training MAE", help="Generally, the lower the MAE, the better the model. It's a good measure of how well the model is performing.",
                       value=training_metrics[1])
        cols[2].metric(label="Training R^2", help="Generally, the higher the R^2, the better the model. It's a good measure of how well the model is performing.",
                       value=training_metrics[2])

        cols[0].metric(label="Testing RMSE", help="Generally, the lower the RMSE, the better the model. It's a good measure of how well the model is performing.",
                       value=testing_metrics[0])
        cols[1].metric(label="Testing MAE", help="Generally, the lower the MAE, the better the model. It's a good measure of how well the model is performing.",
                       value=testing_metrics[1])
        cols[2].metric(label="Testing R^2", help="Generally, the higher the R^2, the better the model. It's a good measure of how well the model is performing.",
                       value=testing_metrics[2])


st.set_page_config(page_title="ML Assignment 2",
                   page_icon="🤖", layout="wide")
raw_df, df = load_data()
rng = np.random.RandomState(0)

st.sidebar.title("ML Assignment 2")
option = st.sidebar.selectbox("Select the scenario", [
                              "Problem 2: Listing Prices"])

if option == "Problem 2: Listing Prices":
    st.sidebar.subheader("Listing Prices")
    st.sidebar.markdown("""
    This problem revolves around the scenario where we want to predict the price of a listing based on the features of the listing.
                        
    The available factors include:
      - ID and name of the listing
      - ID and name of the host
      - Neighbourhood group and neighbourhood
      - Latitude and longitude
      - The type of room available (i.e., private room, shared room, or the entire home)
      - The minimum number of nights that the listing can be booked for
      - The number of reviews and the last review date
      - The number of reviews per month
      - The number of listings the host has
      - The number of days annually the listing is available
""")

explore_tab, models_tab, predict_tab = st.tabs(
    [":rocket: Explore", ":brain: Models", ":question: Predict"])

# Explore tab
explore_tab.title("Exploring the dataset")
explore_tab.markdown(
    """
    The dataset has been cleaned and transformed. The following processes have been done to the dataset:

      - Handling any missing values
      - Handling any outliers (in this case, they were trimmed)
    
    The `price` column is the target feature, where we'll try to predict the price of the listing based on the other features.
    """)
explore_tab.dataframe(df, use_container_width=True)
explore_tab.markdown(
    """
    A hint that was given was to split the dataset into smaller datasets based on specific factors, then to create specific models for each of these datasets. This is to ensure that the models are more accurate and can predict the price of the listing more accurately, instead of using a single model for all the listings.

    In this assignment, I've taken a look at the `neighbourhood_group` and `neighbourhood` columns and created specific models for these. In the interest of time, I've only created models for:
      - Listings with a Central Region `neighbourhood_group`
      - Listings for the Kallang `neighbourhood`
    """)

# Splitting the dataset into neighborhood groups and encoding the neighborhood categorical feature
neighborhood_groups = [group for group in df.groupby(
    "neighbourhood_group", as_index=False)]
neighborhoods = [group for group in df.groupby(
    "neighbourhood", as_index=False)]

for group in neighborhood_groups:
    group_df = group[1]
    group_df["neighbourhood"] = OrdinalEncoder(categories=[
                                               group_df["neighbourhood"].unique()]).fit_transform(group_df[["neighbourhood"]])
    group_df.drop(["neighbourhood_group", "reviews_per_month",
                  "latitude", "longitude"], axis=1, inplace=True)

for group in neighborhoods:
    group_df = group[1]
    group_df.drop(["neighbourhood_group", "neighbourhood",
                  "reviews_per_month", "latitude", "longitude"], axis=1, inplace=True)

central_df = neighborhood_groups[0][1]
kallang_df = neighborhoods[15][1]

cols = explore_tab.columns(2)
cols[0].header("Central Region listings")
cols[0].dataframe(central_df, use_container_width=True)

features = ["neighbourhood", "room_type",
            "minimum_nights", "number_of_reviews", "availability_365"]
for feature in features:
    fig = px.bar(central_df, x=feature, y="price")
    cols[0].plotly_chart(fig, use_container_width=True)

cols[1].header("Kallang listings")
cols[1].dataframe(kallang_df, use_container_width=True)

features = ["room_type", "minimum_nights",
            "number_of_reviews", "availability_365"]
for feature in features:
    fig = px.bar(kallang_df, x=feature, y="price")
    cols[1].plotly_chart(fig, use_container_width=True)

# Models tab
models = []
if option == "Problem 2: Listing Prices":
    models = [
        {"name": "Linear regression", "description": "Linear regression is a simple and commonly used algorithm for predicting numerical values. It assumes a linear relationship between the input features and the target variable. The algorithm finds the best-fit line that minimizes the difference between the predicted and actual values. It is suitable for problems where the relationship between the features and the target variable is approximately linear.", "short_name": "lr"},
        {"name": "Support vector machine", "description": "Support vector machine (SVM) is a powerful algorithm used for both classification and regression tasks. It works by finding the optimal hyperplane that separates the data points into different classes or predicts the numerical values. SVM aims to maximize the margin between the support vectors (data points closest to the decision boundary) and the hyperplane. It is effective in handling complex datasets with non-linear relationships.", "short_name": "svm"},
        {"name": "Multilayered perceptron",
            "description": "Multilayered perceptron (MLP) is a type of artificial neural network that consists of multiple layers of interconnected nodes called neurons. Each neuron applies a non-linear activation function to the weighted sum of its inputs. MLP is capable of learning complex patterns and relationships in the data. It is widely used for various tasks, including classification and regression.", "short_name": "mlp"},
        {"name": "AdaBoost", "description": "AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak classifiers to create a strong classifier. Each weak classifier is trained on a subset of the data, and the algorithm assigns higher weights to misclassified samples in each iteration. AdaBoost iteratively improves the performance by focusing on the difficult samples. It is particularly effective in handling imbalanced datasets and can be used for both classification and regression problems.", "short_name": "ab"}]
models_tab.title("Models used for prediction")
lr_tab, svm_tab, mlp_tab, ab_tab = models_tab.tabs(
    [model["name"] for model in models])

# Setting up the training and testing data for the models
central_X = central_df.drop("price", axis=1)
central_y = central_df["price"]
central_X_train, central_X_test, central_y_train, central_y_test = train_test_split(
    central_X, central_y, test_size=0.3, random_state=rng)

kallang_X = kallang_df.drop("price", axis=1)
kallang_y = kallang_df["price"]
kallang_X_train, kallang_X_test, kallang_y_train, kallang_y_test = train_test_split(
    kallang_X, kallang_y, test_size=0.3, random_state=rng)

for index, tab in enumerate([lr_tab, svm_tab, mlp_tab, ab_tab]):
    tab.title(models[index]["name"])
    tab.markdown(models[index]["description"])

    central_model = load_model(f"central_{models[index]['short_name']}.pkl")
    kallang_model = load_model(f"kallang_{models[index]['short_name']}.pkl")

    tab.header("Central Region listings")
    display_metrics(tab, central_model, central_X_train,
                    central_y_train, central_X_test, central_y_test)

    tab.header("Kallang listings")
    display_metrics(tab, kallang_model, kallang_X_train,
                    kallang_y_train, kallang_X_test, kallang_y_test)

# Predict tab
predict_tab.title("Predicting the output")
predict_tab.markdown(
    "Use the form below to give the models a try. Just input in some values and see what the model predicts!")

form = predict_tab.form(key="prediction_form")
form.selectbox("Model", [model["name"] for model in models])

if option == "Problem 2: Listing Prices":
    cols = form.columns(5)

    selected_neighborhood_group = cols[0].selectbox("Neighbourhood group",
                                                    ["Central Region"])
    selected_neighborhood = cols[0].selectbox("Neighbourhood", df[df["neighbourhood_group"] ==
                                                                  "Central Region"]["neighbourhood"].unique())
    selected_room_type = cols[1].selectbox("Room type", ["Private room",
                                                         "Shared room", "Entire home/apt"])
    minimum_nights = cols[2].number_input("Minimum nights", 1)
    annual_availability = cols[3].number_input("Annual availability (days)", 0)
    number_of_reviews = cols[4].number_input("Number of reviews", 0)

    if form.form_submit_button(":question: Predict"):
        # Manually encodes the minimum nights and annual availability using min-max scaling
        minimum_nights = (minimum_nights - raw_df["minimum_nights"].min()) / \
            (raw_df["minimum_nights"].max() - raw_df["minimum_nights"].min())
        annual_availability = (annual_availability - raw_df["availability_365"].min()) / \
            (raw_df["availability_365"].max() -
             raw_df["availability_365"].min())
        number_of_reviews = (number_of_reviews - raw_df["number_of_reviews"].min()) / \
            (raw_df["number_of_reviews"].max() -
             raw_df["number_of_reviews"].min())

        # Encodes the neighborhood and room type using indexing
        encoded_neighborhood = OrdinalEncoder(categories=[
            df[df["neighbourhood_group"] == selected_neighborhood_group]["neighbourhood"].unique()]).fit_transform([[selected_neighborhood]])[0][0]
        selected_room_type = OrdinalEncoder(categories=[
                                            ["Entire home/apt", "Private room", "Shared room"]]).fit_transform([[selected_room_type]])[0][0]

        # Forms a single-row dataframe to predict the price
        input_df = pd.DataFrame({"neighbourhood": [encoded_neighborhood], "room_type": [
                                selected_room_type], "minimum_nights": [minimum_nights], "number_of_reviews": [number_of_reviews], "availability_365": [annual_availability]})

        # Predicts the price using the selected model
        if selected_neighborhood == "Kallang":
            model = load_model(f"kallang_{models[0]['short_name']}.pkl")
            input_df.drop("neighbourhood", axis=1, inplace=True)
        else:
            model = load_model(f"central_{models[0]['short_name']}.pkl")

        prediction = model.predict(input_df)
        predict_tab.metric(label="Predicted price",
                           value="$" + str(round(prediction[0], 2)))
