from flask import Flask, render_template_string, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for caching
movies_df = None
similarity_matrix = None

# Load dataset from CSV
def load_data():
    """Load and preprocess movie data with error handling"""
    try:
        if not os.path.exists("dataset.csv"):
            logger.error("Dataset file 'dataset.csv' not found")
            return None
            
        df = pd.read_csv("dataset.csv")
        
        # Check if required columns exist
        required_columns = ['Title', 'Overview', 'Poster_Url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None

        # Rename columns to lowercase for consistency
        df.rename(columns={
            'Title': 'title',
            'Overview': 'overview',
            'Poster_Url': 'poster_url'
        }, inplace=True)

        # Drop rows with missing title or overview
        initial_count = len(df)
        df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
        final_count = len(df)
        
        if final_count == 0:
            logger.error("No valid movies found after cleaning data")
            return None
            
        logger.info(f"Loaded {final_count} movies (removed {initial_count - final_count} invalid entries)")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

# Compute similarity matrix
@lru_cache(maxsize=1)
def compute_similarity(df_hash):
    """Compute similarity matrix with caching"""
    try:
        global movies_df
        if movies_df is None:
            return None
            
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))
        similarity_matrix = cosine_similarity(tfidf_matrix)
        logger.info("Similarity matrix computed successfully")
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return None

# Recommend similar movies
def recommend(movie_title, df, similarity_matrix):
    """Get movie recommendations with error handling"""
    try:
        # Find movie index
        movie_matches = df[df['title'].str.lower() == movie_title.lower()]
        
        if movie_matches.empty:
            logger.warning(f"Movie '{movie_title}' not found")
            return []
            
        index = movie_matches.index[0]
        
        # Get similarity scores
        distances = list(enumerate(similarity_matrix[index]))
        sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        
        recommendations = []
        for i, score in sorted_distances:
            movie_data = df.iloc[i]
            recommendations.append({
                'title': movie_data['title'],
                'overview': movie_data['overview'][:150] + "..." if len(movie_data['overview']) > 150 else movie_data['overview'],
                'poster_url': movie_data['poster_url'] if pd.notna(movie_data['poster_url']) else "https://via.placeholder.com/200x300?text=No+Image",
                'similarity_score': round(score, 3)
            })
            
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []

# Initialize data on startup
def initialize_app():
    """Initialize the application data"""
    global movies_df, similarity_matrix
    
    logger.info("Initializing movie recommender app...")
    movies_df = load_data()
    
    if movies_df is not None:
        # Create a hash for caching
        df_hash = hash(str(movies_df.values.tobytes()))
        similarity_matrix = compute_similarity(df_hash)
        
        if similarity_matrix is not None:
            logger.info("App initialized successfully")
            return True
    
    logger.error("Failed to initialize app")
    return False

# HTML Template with mobile-optimized CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Movie Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Toffee, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px 15px;
            min-height: 100vh;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            padding: 25px 20px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.2rem;
            color: #4a5568;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            color: #718096;
            font-size: 1rem;
        }

        .search-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px 20px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4a5568;
            font-size: 1.1rem;
        }

        select, input {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background: white;
            transition: all 0.3s ease;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }

        select {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 12px center;
            background-repeat: no-repeat;
            background-size: 16px;
            padding-right: 40px;
        }

        .btn {
            width: 100%;
            padding: 15px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        .btn:active {
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 1.1rem;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results {
            display: none;
        }

        .results-header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 20px 20px 0 0;
            text-align: center;
            backdrop-filter: blur(10px);
        }

        .results-header h2 {
            color: #4a5568;
            font-size: 1.5rem;
            margin-bottom: 5px;
        }

        .results-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 0 0 20px 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }

        .movie-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .movie-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }

        .movie-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }

        .movie-poster {
            width: 100%;
            max-width: 150px;
            height: auto;
            border-radius: 10px;
            margin: 0 auto 15px;
            display: block;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        .movie-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 10px;
            text-align: center;
        }

        .movie-overview {
            color: #718096;
            font-size: 0.9rem;
            line-height: 1.5;
            text-align: justify;
        }

        .similarity-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 10px;
        }

        .error {
            background: rgba(254, 226, 226, 0.95);
            color: #c53030;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 4px solid #fc8181;
            backdrop-filter: blur(10px);
        }

        .no-data {
            text-align: center;
            padding: 40px 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }

        .no-data h3 {
            color: #4a5568;
            margin-bottom: 10px;
        }

        .no-data p {
            color: #718096;
        }

        /* Responsive Design */
        @media (min-width: 480px) {
            .container {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2.5rem;
            }
            
            .movie-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (min-width: 768px) {
            .container {
                max-width: 800px;
                padding: 40px 30px;
            }
            
            .movie-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (min-width: 1024px) {
            .movie-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        /* Touch-friendly interactions */
        @media (hover: none) {
            .movie-card:hover {
                transform: none;
            }
            
            .btn:hover {
                transform: none;
            }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            body {
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Movie Recommender</h1>
            <p>Discover your next favorite movie based on AI-powered recommendations this app created by ASIF</p>
        </div>

        {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
        {% elif not movies %}
        <div class="no-data">
            <h3>🎭 No Movies Available</h3>
            <p>Please make sure the dataset.csv file is properly loaded.</p>
        </div>
        {% else %}
        <div class="search-container">
            <form id="recommendForm">
                <div class="form-group">
                    <label for="movie-select">🎞️ Select a Movie:</label>
                    <select id="movie-select" name="movie" required>
                        <option value="">Choose a movie...</option>
                        {% for movie in movies %}
                        <option value="{{ movie }}">{{ movie }}</option>
                        {% endfor %}
                    </select>
                </div>
                <button type="submit" class="btn" id="recommend-btn">
                    ✨ Get Recommendations
                </button>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            Finding perfect matches for you...
        </div>

        <div class="results" id="results">
            <div class="results-header">
                <h2>🎯 You May Also Like</h2>
                <p>Based on your selection</p>
            </div>
            <div class="results-container">
                <div class="movie-grid" id="movie-grid">
                    <!-- Recommendations will be inserted here -->
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('recommendForm');
            const loadingDiv = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');
            const movieGrid = document.getElementById('movie-grid');
            const recommendBtn = document.getElementById('recommend-btn');

            if (form) {
                form.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const movieSelect = document.getElementById('movie-select');
                    const selectedMovie = movieSelect.value;
                    
                    if (!selectedMovie) {
                        alert('Please select a movie first!');
                        return;
                    }

                    // Show loading state
                    loadingDiv.style.display = 'block';
                    resultsDiv.style.display = 'none';
                    recommendBtn.disabled = true;
                    recommendBtn.textContent = 'Getting Recommendations...';

                    try {
                        const response = await fetch('/recommend', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ movie: selectedMovie })
                        });

                        const data = await response.json();

                        if (data.error) {
                            throw new Error(data.error);
                        }

                        // Hide loading
                        loadingDiv.style.display = 'none';
                        
                        // Display results
                        displayRecommendations(data.recommendations);
                        resultsDiv.style.display = 'block';
                        
                        // Scroll to results
                        resultsDiv.scrollIntoView({ behavior: 'smooth' });

                    } catch (error) {
                        console.error('Error:', error);
                        loadingDiv.style.display = 'none';
                        alert('Error getting recommendations: ' + error.message);
                    } finally {
                        recommendBtn.disabled = false;
                        recommendBtn.textContent = '✨ Get Recommendations';
                    }
                });
            }

            function displayRecommendations(recommendations) {
                if (!recommendations || recommendations.length === 0) {
                    movieGrid.innerHTML = '<div class="no-data"><h3>No recommendations found</h3><p>Please try another movie.</p></div>';
                    return;
                }

                movieGrid.innerHTML = recommendations.map(movie => `
                    <div class="movie-card">
                        <img src="${movie.poster_url}" alt="${movie.title}" class="movie-poster" 
                             onerror="this.src='https://via.placeholder.com/200x300?text=No+Image'">
                        <div class="movie-title">🎥 ${movie.title}</div>
                        <div class="movie-overview">${movie.overview}</div>
                        <span class="similarity-score">Match: ${(movie.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page route"""
    global movies_df
    
    if movies_df is None:
        return render_template_string(HTML_TEMPLATE, 
                                    error="Failed to load movie data. Please check if dataset.csv exists and is properly formatted.",
                                    movies=None)
    
    if movies_df.empty:
        return render_template_string(HTML_TEMPLATE, 
                                    error="No valid movies found in the dataset.",
                                    movies=None)
    
    movies_list = sorted(movies_df['title'].tolist())
    return render_template_string(HTML_TEMPLATE, movies=movies_list, error=None)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint for getting movie recommendations"""
    global movies_df, similarity_matrix
    
    try:
        data = request.get_json()
        
        if not data or 'movie' not in data:
            return jsonify({'error': 'Movie name is required'}), 400
            
        movie_name = data['movie'].strip()
        
        if not movie_name:
            return jsonify({'error': 'Movie name cannot be empty'}), 400
            
        if movies_df is None:
            return jsonify({'error': 'Movie database not available'}), 500
            
        if similarity_matrix is None:
            return jsonify({'error': 'Recommendation system not ready'}), 500
        
        recommendations = recommend(movie_name, movies_df, similarity_matrix)
        
        if not recommendations:
            return jsonify({'error': f'No recommendations found for "{movie_name}"'}), 404
            
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        print("🎬 Movie Recommender App is starting...")
        print("📱 Optimized for mobile devices")
        print("🚀 Running on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize the application")
        print("Please ensure 'dataset.csv' exists with columns: Title, Overview, Poster_Url")