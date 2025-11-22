import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Set plot style
plt.style.use('ggplot')

def load_and_merge_data(credits_path, movies_path):
    """Loads and merges the credits and movies datasets."""
    print("Loading data...")
    try:
        credits = pd.read_csv(credits_path)
        movies = pd.read_csv(movies_path)
    except FileNotFoundError:
        print("Error: CSV files not found. Please ensure 'tmdb_5000_credits.csv' and 'tmdb_5000_movies.csv' are in the directory.")
        return None

    # Merge on ID (renaming movie_id in credits to match id in movies)
    credits.columns = ['id', 'tittle', 'cast', 'crew']
    movies = movies.merge(credits, on='id')
    
    print(f"Data merged. Shape: {movies.shape}")
    return movies

def parse_json_features(df):
    """Parses stringified JSON columns into python objects."""
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)
    return df

def get_director(x):
    """Extracts the director's name from the crew list."""
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    """Returns the top 3 elements from a list (for cast, keywords, genres)."""
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def prepare_features_for_regression(df):
    """Prepares numerical and categorical features for regression."""
    print("Preparing features for Regression Model...")
    
    # Create a copy to avoid SettingWithCopy warnings
    data = df.copy()
    
    # 1. Extract basic features
    data['director'] = data['crew'].apply(get_director)
    data['genre_list'] = data['genres'].apply(lambda x: [i['name'] for i in x])
    
    # 2. Filter for valid numerical data
    # We use budget, runtime, and vote_count as predictors
    # Note: We filter out movies with 0 budget or 0 runtime as they are likely missing data
    data = data[(data['budget'] > 0) & (data['runtime'] > 0)]
    
    # 3. One-Hot Encode Genres
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(data['genre_list']), columns=mlb.classes_, index=data.index)
    
    # 4. Combine features
    # We will use 'budget', 'runtime', 'vote_count', 'popularity' + Genres
    numerical_features = data[['budget', 'runtime', 'vote_count', 'popularity']]
    X = pd.concat([numerical_features, genres_encoded], axis=1)
    y = data['vote_average']
    
    return X, y

def train_regression_model(X, y):
    """Trains a Random Forest Regressor to predict movie ratings."""
    print("\n--- Training Regression Model ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print("\nTop 10 Important Features for Rating Prediction:")
    print(feature_importance)
    
    return model

def create_soup(x):
    """Combines keywords, cast, director, and genres into a single string (soup) for vectorization."""
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Extract basic lists
    keywords = [i['name'] for i in x['keywords']]
    cast = [i['name'] for i in x['cast'][:3]] # Top 3 actors
    genres = [i['name'] for i in x['genres']]
    director = get_director(x['crew'])
    
    # Apply cleaning
    keywords = clean_data(keywords)
    cast = clean_data(cast)
    genres = clean_data(genres)
    director = clean_data(director)
    
    # Combine
    return ' '.join(keywords) + ' ' + ' '.join(cast) + ' ' + ' '.join(director) + ' ' + ' '.join(genres)

def build_recommender_system(df):
    """Builds a content-based recommendation system."""
    print("\n--- Building Recommendation System ---")
    
    # Preprocessing for content-based filtering
    df['soup'] = df.apply(create_soup, axis=1)
    
    # Use CountVectorizer to create a count matrix
    # using stop_words='english' to remove common words
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    
    # Compute Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Reset index to make sure we can query by title
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title_x']).drop_duplicates()
    
    return cosine_sim, indices, df

def get_recommendations(title, cosine_sim, indices, df):
    """Returns a list of top 10 similar movies."""
    try:
        idx = indices[title]
    except KeyError:
        return [f"Movie '{title}' not found in dataset."]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort based on scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of the 10 most similar movies (ignoring the first one which is itself)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title_x'].iloc[movie_indices].tolist()

def main():
    # File paths
    credits_file = 'tmdb_5000_credits.csv'
    movies_file = 'tmdb_5000_movies.csv'
    
    # 1. Load and Parse Data
    df = load_and_merge_data(credits_file, movies_file)
    if df is None: return
    
    df = parse_json_features(df)
    
    # 2. Rating Prediction (Regression)
    # Feature Engineering specifically for Regression
    X, y = prepare_features_for_regression(df)
    rf_model = train_regression_model(X, y)
    
    # 3. Recommendation System
    cosine_sim, indices, df_rec = build_recommender_system(df)
    
    # Example Recommendations
    test_movie = 'The Dark Knight Rises'
    print(f"\nGenerating recommendations for: {test_movie}")
    recommendations = get_recommendations(test_movie, cosine_sim, indices, df_rec)
    
    print("Top 10 Recommended Movies:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

    # Example 2
    test_movie_2 = 'Avatar'
    print(f"\nGenerating recommendations for: {test_movie_2}")
    recommendations_2 = get_recommendations(test_movie_2, cosine_sim, indices, df_rec)
    
    print("Top 10 Recommended Movies:")
    for i, movie in enumerate(recommendations_2, 1):
        print(f"{i}. {movie}")

if __name__ == "__main__":
    main()