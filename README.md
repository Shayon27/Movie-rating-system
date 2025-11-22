Movie Rating Prediction & Recommendation System

📜 Project Description

This project analyzes the TMDB 5000 Movie Dataset to build two intelligent systems: a Rating Predictor and a Movie Recommendation Engine.

Rating Predictor: A machine learning model that estimates a movie's average rating based on features like budget, runtime, and genres. It helps identify the key factors that influence audience reception.

Recommendation System: A content-based filtering engine that suggests movies similar to a user's favorite film by analyzing metadata such as cast, crew, keywords, and genres.

🛠️ Skills & Technologies Used

Python: Core programming language.

Data Analysis (Pandas & NumPy): Data manipulation, cleaning, and merging complex datasets.

Machine Learning (Scikit-Learn): * Regression: Random Forest Regressor for predicting continuous variables (ratings).

Model Evaluation: Using RMSE, MAE, and R² scores.

Natural Language Processing (NLP): Feature extraction using CountVectorizer to process text data (keywords, genres, cast).

Recommendation Algorithms: Implementation of Cosine Similarity for content-based filtering.

Data Visualization: Matplotlib and Seaborn for exploratory data analysis.

🚀 How It Works

Data Processing: Parses stringified JSON columns (like genres and crew) to extract usable features.

Prediction: Uses a Random Forest model to predict user ratings based on budget, runtime, popularity, and genre.

Recommendation: Creates a "metadata soup" (combined text features) for each movie and calculates similarity scores to find the closest matches to a given title.
