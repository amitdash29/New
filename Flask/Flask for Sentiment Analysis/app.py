import os
from flask import Flask, render_template, request, send_file
import pandas as pd
from textblob import TextBlob
from io import BytesIO

# Initialize Flask application
app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check file type by extension
def allowed_file(filename):
    # This function allows any file type to be uploaded.
    return '.' in filename

# Function to perform sentiment analysis
def analyze_sentiment(text):
    if pd.isnull(text):  # Handle missing values
        return None
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a score between -1 (negative) and 1 (positive)

# Function to process the file and apply sentiment analysis
def process_file(file_path, file_extension):
    # Load the dataset based on file type
    if file_extension in ['xlsx', 'xls']:
        data = pd.read_excel(file_path)
    elif file_extension == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension == 'json':
        data = pd.read_json(file_path)
    elif file_extension == 'txt':
        data = pd.read_csv(file_path, sep='\t')  # Assuming tab-separated text file
    else:
        raise ValueError("Unsupported file type")

    # Define column renaming
    column_rename_map = {
        'Elaborate on your contributions to the success of your team?': 'contribution_to_team',
        'How would you rate your ability to proactively manage your own career development': 'career_dev_rate',
        'Please elaborate on the ability to proactively manage your own career development': 'career_dev_text',
        'How committed are you in embracing the culture and values within the organization?': 'culture_value_rate',
        'Elaborate on the commitment in embracing the culture and values within the organization': 'culture_value_text',
        'How would you rate your own growth mindset when facing challenges and failures in the workplace?': 'challenges_failures_rate',
        'Elaborate your own growth mindset when facing challenges and failures in the workplace?': 'challenges_failures_text',
        'How would you rate your communication and Collaboration with the team?': 'comm_collab_rate',
        'Elaborate on your ability to communication and Collaborate with the team? ': 'comm_collab_text',
        'How would you rate yourself in terms of inspiring the team?': 'inspiring_team_rate',
        'Elaborate on your ability, in terms of inspiring the team. ': 'inspiring_team_text',
        'How would you rate yourself in terms of delivery of work and  your sense of urgency in completing the task?': 'task_delivery_rate',
        'How would you ensure timely delivery of the work and cultivate a sense of urgency in completing the task?': 'task_delivery_text',
        'How would you rate yourself in effectiveness in leading change within the department?': 'leading_change_rate',
        'Elaborate yourself in effectiveness in leading change within the department': 'leading_change_text',
        'How would you rate yourself in terms of innovation and your ability to generate new ideas or approaches within your role?': 'innovation_ideas_rate',
        'Elaborate on your ability to generate new ideas or approaches within your role?': 'innovation_ideas_text',
        'How would your rate your ability to think outside the box when creating designs and providing solutions?': 'outside_box_rate',
        'Elaborate on your ability to think outside the box when creating designs and providing solutions?': 'outside_box_text',
        'What are your achievements in the last quarter(Oct-Dec)?': 'achivements'
    }

    # Rename columns using the dictionary
    data.rename(columns=column_rename_map, inplace=True)

    string_columns = data.select_dtypes(include=['object']).columns

    threshold = len(string_columns) / 2

    df = data[data[string_columns].isnull().sum(axis=1) < threshold]
    df.fillna('0', inplace=True)

    print(df)


    # Function to perform sentiment analysis
    def analyze_sentiment(text):
        if pd.isnull(text):  # Handle missing values
            return None
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Returns a score between -1 (negative) and 1 (positive)

    # Apply sentiment analysis on all feedback columns
    feedback_columns = [col for col in df.columns if '_text' in col]

    for col in feedback_columns:
        sentiment_col = col.replace('_text', '_sentiment')
        df[sentiment_col] = df[col].apply(analyze_sentiment)

    # Derive sentiment labels from polarity scores
    # Sentiment rules: Negative (<0), Neutral (0), Positive (>0)
    for col in feedback_columns:
        sentiment_label_col = col.replace('_text', '_label')
        df[sentiment_label_col] = df[col.replace('_text', '_sentiment')].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
        )

    # Generate true labels from ratings (assumption: >=4 is positive, <=2 is negative, else neutral)
    rating_columns = [col for col in df.columns if '_rate' in col]
    for col in rating_columns:
        true_label_col = col.replace('_rate', '_true_label')
        df[true_label_col] = df[col].apply(
            lambda x: 'positive' if x >= 4 else ('negative' if x <= 3 else 'neutral')
        )

    # Collect all predictions and true labels for evaluation
    true_labels = []
    predicted_labels = []

    for col in feedback_columns:
        true_labels.extend(df[col.replace('_text', '_true_label')])
        predicted_labels.extend(df[col.replace('_text', '_label')])

    # Generate the final sentiment analysis column
    def aggregate_sentiment(row):
        sentiments = [row[col.replace('_text', '_label')] for col in feedback_columns]
        if sentiments:
            return max(set(sentiments), key=sentiments.count)
        return 'neutral'

    # final sentiment column addition
    df['Sentiment Analysis'] = df.apply(aggregate_sentiment, axis=1)
    
    # Define a dictionary to map the new column names back to the original names
    reverse_column_rename_map = {
        'contribution_to_team': 'Elaborate on your contributions to the success of your team?',
        'career_dev_rate': 'How would you rate your ability to proactively manage your own career development',
        'career_dev_text': 'Please elaborate on the ability to proactively manage your own career development',
        'culture_value_rate': 'How committed are you in embracing the culture and values within the organization?',
        'culture_value_text': 'Elaborate on the commitment in embracing the culture and values within the organization',
        'challenges_failures_rate': 'How would you rate your own growth mindset when facing challenges and failures in the workplace?',
        'challenges_failures_text': 'Elaborate your own growth mindset when facing challenges and failures in the workplace?',
        'comm_collab_rate': 'How would you rate your communication and Collaboration with the team?',
        'comm_collab_text': 'Elaborate on your ability to communication and Collaborate with the team? ',
        'inspiring_team_rate': 'How would you rate yourself in terms of inspiring the team?',
        'inspiring_team_text': 'Elaborate on your ability, in terms of inspiring the team. ',
        'task_delivery_rate': 'How would you rate yourself in terms of delivery of work and  your sense of urgency in completing the task?',
        'task_delivery_text': 'How would you ensure timely delivery of the work and cultivate a sense of urgency in completing the task?',
        'leading_change_rate': 'How would you rate yourself in effectiveness in leading change within the department?',
        'leading_change_text': 'Elaborate yourself in effectiveness in leading change within the department',
        'innovation_ideas_rate': 'How would you rate yourself in terms of innovation and your ability to generate new ideas or approaches within your role?',
        'innovation_ideas_text': 'Elaborate on your ability to generate new ideas or approaches within your role?',
        'outside_box_rate': 'How would your rate your ability to think outside the box when creating designs and providing solutions?',
        'outside_box_text': 'Elaborate on your ability to think outside the box when creating designs and providing solutions?',
        'achivements': 'What are your achievements in the last quarter(Oct-Dec)?'
    }

    # Revert the column renaming using the reverse mapping
    df.rename(columns=reverse_column_rename_map, inplace=True)

    # Drop the temporary columns used for sentiment analysis
    cols_to_drop = [col for col in df.columns if any(suffix in col for suffix in ['_sentiment', '_true_label'])]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Apply color to the 'Sentiment Analysis' column
    def color_sentiment(val):
        color_dict = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
        return f'background-color: {color_dict.get(val, "white")}'

    # Apply the style to the 'Sentiment Analysis' column
    styled_df = df.style.applymap(color_sentiment, subset=['Sentiment Analysis'])

    # Return the processed dataframe
    return styled_df

# Route for uploading file and processing it
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Get file extension
        filename = file.filename
        file_extension = filename.rsplit('.', 1)[1].lower()

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the file based on extension
        df = process_file(file_path, file_extension)

        # Save the processed DataFrame to a CSV in memory
        output = BytesIO()
        df.to_excel(output, engine='openpyxl', index=False)
        output.seek(0)
        
        # Send the file back to the user
        return send_file(output, mimetype='text/xlsx', as_attachment=True, download_name="Sentiment Report.xlsx")

if __name__ == '__main__':
    app.run(debug=True)
