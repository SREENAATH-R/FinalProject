from flask import Flask, render_template, request, redirect, url_for, session, flash
import google.generativeai as genai
import os
import sqlite3
import markdown
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


app = Flask(__name__)
app.config['SECRET_KEY'] = 'Abc@123_Abc@123_'
app.config['UPLOAD_FOLDER'] = 'uploads'
login_manager = LoginManager(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def init_db():
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS chatbot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email))''')
        conn.commit()

init_db()

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

os.environ["API_KEY"] = "AIzaSyDGXhLcMJjKjMY7knXwa_grvN-vTalhdDs"
genai.configure(api_key=os.environ["API_KEY"])

job_listings = []

# Load dataset from CSV
data = pd.read_csv('job_data.csv')

# Fill missing values and ensure data consistency
data['year_of_passing'] = data['year_of_passing'].fillna(0).astype(int)
data['title'] = data['title'].astype(str)
data['company'] = data['company'].astype(str)
data['location'] = data['location'].astype(str)
data['skills'] = data['skills'].astype(str)
data['degree'] = data['degree'].astype(str)
data['branch'] = data['branch'].astype(str)
data['year_of_passing'] = data['year_of_passing'].astype(str)

# Combine relevant features for content-based filtering
data['combined_features'] = (
    data['title'] + " " +
    data['skills'] + " " +
    data['location'] + " " +
    data['degree'] + " " +
    data['branch'] + " " +
    data['year_of_passing']
)

# Vectorize combined features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Rule-Based Matching
def rule_based_score(job, user_skills, user_degree, user_branch, user_location, user_year_of_passing):
    score = 0
    job_skills = ast.literal_eval(job['skills'])
    if user_location and job['location'] == user_location:
        score += 2
    if any(skill in job_skills for skill in user_skills):
        score += 2
    if job['degree'] == user_degree:
        score += 1
    if job['branch'] == user_branch:
        score += 1
    if job['year_of_passing'] == user_year_of_passing:
        score += 1
    return score

# Job Search Function
def search_jobs(user_title, user_skills, user_location, user_degree, user_branch, user_year_of_passing):
    user_input = f"{user_title} {' '.join(user_skills)} {user_location} {user_degree} {user_branch} {user_year_of_passing}"
    user_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    content_based_indices = cosine_similarities.argsort()[-20:][::-1]

    rule_scores = data.apply(lambda job: rule_based_score(job, user_skills, user_degree, user_branch, user_location, user_year_of_passing), axis=1)
    rule_based_indices = rule_scores.nlargest(20).index

    combined_indices = list(set(content_based_indices) | set(rule_based_indices))
    combined_indices = [int(i) for i in combined_indices if isinstance(i, (int, np.integer))]

    if combined_indices:
        recommendations = data.iloc[combined_indices].copy()
        recommendations.reset_index(drop=True, inplace=True)
    else:
        recommendations = pd.DataFrame()

    if not recommendations.empty:
        recommendations['hybrid_score'] = (
            0.5 * recommendations.index.isin(content_based_indices).astype(int) +
            0.5 * recommendations.index.isin(rule_based_indices).astype(int)
        )
        recommendations = recommendations.sort_values(by='hybrid_score', ascending=False).head(10)

    return recommendations[['title', 'company', 'location', 'skills', 'degree', 'branch', 'year_of_passing', 'hybrid_score']] if not recommendations.empty else pd.DataFrame()


@app.route('/')
def home():
    return render_template('base.html')

@app.route('/main')
@login_required
def main():
    user_email = session['email']

    # Fetch conversation history from the database
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT message, response FROM chatbot WHERE email = ? ORDER BY timestamp ASC', (user_email,))
        conversation_history = cursor.fetchall()

    # Convert tuples to a list of dictionaries for easier access in the template
    conversation_history = [{'message': entry[0], 'response': entry[1]} for entry in conversation_history]

    return render_template('main.html', conversation_history=conversation_history, show_chatbot=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        with sqlite3.connect('app.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT username, password FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            session['email'] = email
            session['username'] = user[0]
            login_user(User(email))
            return redirect(url_for('main'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        with sqlite3.connect('app.db') as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                               (username, email, generate_password_hash(password)))
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists', 'danger')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('email', None)
    session.pop('username', None)
    logout_user()
    return redirect(url_for('login'))
@app.route('/get_recommendations/<username>', methods=['POST'])
@login_required
def get_recommendations(username):
    # Retrieve user input from form
    user_title = request.form.get('title')
    education = request.form.get('education')
    degree = request.form.get('degree')
    branch = request.form.get('branch')
    year_of_passing = request.form.get('year_of_passing')
    skills = request.form.get('skills')
    user_location = request.form.get('location')
    user_email = session['email']

    # Process skills from the user input
    user_skills = skills.lower().split(", ")

    # Manual filtering based on user skills
    recommended_jobs = []
    for job in job_listings:
        job_skills = job['skills'].lower().split(", ")
        if any(skill in job_skills for skill in user_skills):
            recommended_jobs.append(job)

    # Get enhanced recommendations using the search_jobs function
    search_results = search_jobs(
        user_title=user_title,
        user_skills=user_skills,
        user_location=user_location,
        user_degree=degree,
        user_branch=branch,
        user_year_of_passing=year_of_passing
    )

    # Combine both recommendation sets, ensuring uniqueness and prioritizing search_results
    all_recommended_jobs = pd.concat([pd.DataFrame(search_results), pd.DataFrame(recommended_jobs)]).drop_duplicates().to_dict(orient='records')

    # Generate career improvement advice using the Gemini API
    advice_prompt = f"The user has the following qualifications: Degree: {degree}, Branch: {branch}, Skills: {skills}. Suggest some improvements to help them achieve their career goal."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(advice_prompt)

    # Convert advice response to HTML
    advice_html = markdown.markdown(response.text)

        # Certification recommendation prompt
    certification_prompt = f"The user has the following skills: {skills} and background: Degree in {degree}, Branch: {branch}. Suggest relevant certifications with links for each to enhance their skills in their career field."
    certification_response = model.generate_content(certification_prompt)

    # Parse and format the certification recommendations as HTML with links
    certification_html = markdown.markdown(certification_response.text)

    # Store advice in conversation history
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chatbot (email, message, response) VALUES (?, ?, ?)', 
                       (user_email, "Advice Request", advice_html))
        conn.commit()

    # Fetch updated conversation history
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT message, response FROM chatbot WHERE email = ? ORDER BY timestamp ASC', (user_email,))
        conversation_history = cursor.fetchall()

    # Convert conversation history to a list of dictionaries
    conversation_history = [{'message': entry[0], 'response': entry[1]} for entry in conversation_history]

    # Render the main template with recommendations and chatbot conversation
    return render_template('main.html', jobs=all_recommended_jobs, advice=advice_html,certifications=certification_html, conversation_history=conversation_history, show_chatbot=True)

@app.route('/chatbot_interaction', methods=['POST'])
@login_required
def chatbot_interaction():
    user_input = request.form.get('chatbot_input')
    user_email = session['email']

    # Incorporate user's career recommendations into the chatbot context
    last_advice = "No previous recommendation."
    
    # Get the last conversation entry if it exists
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT response FROM chatbot WHERE email = ? ORDER BY timestamp DESC LIMIT 1', (user_email,))
        last_advice_entry = cursor.fetchone()
        if last_advice_entry:
            last_advice = last_advice_entry[0]

    # Prepare the user's query prompt
    user_query_prompt = f"Based on this career advice: {last_advice}, respond to this user query: {user_input}"

    # Send the user's input to the chatbot model
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(user_query_prompt)
    
    # Convert response from markdown to HTML
    bot_response_html = markdown.markdown(response.text)

    # Store the new input and response in the database
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chatbot (email, message, response) VALUES (?, ?, ?)', 
                       (user_email, user_input, bot_response_html))
        conn.commit()

    # Fetch updated conversation history from the database
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT message, response FROM chatbot WHERE email = ? ORDER BY timestamp ASC', (user_email,))
        conversation_history = cursor.fetchall()

    # Convert tuples to a list of dictionaries for easier access in the template
    conversation_history = [{'message': entry[0], 'response': entry[1]} for entry in conversation_history]

    return render_template('main.html', conversation_history=conversation_history, show_chatbot=True)

if __name__ == '__main__':
    app.run(debug=True)
