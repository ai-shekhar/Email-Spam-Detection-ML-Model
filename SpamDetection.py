import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Loading the dataset
data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)
data.isnull().sum()
# print(data.shape)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
# print(data.head())

# Splitting the dataset into training and testing sets
mesg = data['Message']
cat = data['Category']
X_train, X_test, y_train, y_test = train_test_split(mesg, cat, test_size=0.2, random_state=42)

# Vectorizing the text data
cv = CountVectorizer()
features = cv.fit_transform(X_train)

# Creating the model
model = MultinomialNB()
model.fit(features, y_train) # Training the model

# Evaluating (Testng) the model
features_test = cv.transform(X_test)
# print("Model Accuracy: ", model.score(features_test, y_test))

# Making a predicton
def predict(message):
    message = cv.transform([message]).toarray()
    result = model.predict(message)
    return result
# print("The message is:  ", result)

st.title("Spam Detection App")

# Initialize session state for the text input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def clear_text():
    st.session_state.user_input = ""

# Text input
input_mesg = st.text_input("Enter the message:", value=st.session_state.user_input, key="user_input")

# Layout for buttons only
col1, col2, col3 = st.columns([1, 1, 4]) 

with col1:
    predict_clicked = st.button("Predict")

with col2:
    st.button("Clear Text", on_click=clear_text)

# --- Logic moved OUTSIDE of columns to use full width ---
if predict_clicked:
    if input_mesg.strip() != "":
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        
        output = predict(input_mesg)
        st.success(f"The message is: {output[0]}")
    else:
        # This will now appear in a single line across the screen
        st.warning("Please enter a message first!")