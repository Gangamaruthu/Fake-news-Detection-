from flask import Flask , request, render_template, redirect , url_for
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    msg =''
    output = ""    
    if request.method == 'POST':
        news = request.form['news']
        import numpy as np
        import pandas as pd
        df= pd.read_csv("static/Fake2.csv", nrows= 100)
        df.head()

        df = df[["title", "text", "result"]]

        print(df.head())
        print(df.shape)

        df.info()

        df['result'] = df['result'].replace({'Fake':0 , 'TRUE':1})

        df.info()

        df.head()

        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords

        print("Title:", df.iloc[3]["title"])
        print("Text:", df.iloc[3]["text"])

        # Corrected clean_html function to remove actual HTML tags
        def clean_html(value):
        # Use regex to remove HTML tags
            clean = re.compile('<.*?>')
            return re.sub(clean , '' , value)

        # Corrected remove_special function to replace non-alphanumeric characters with single spaces
        def remove_special(value):
        # Replace non-alphanumeric characters (excluding spaces) with a space
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', value)
            # Replace multiple spaces with a single space and strip leading/trailing spaces
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            return cleaned_text

        def convert_lower(value):
            return value.lower()

        def remove_stopwords(value):
            x=[]
            for i in value.split():
                if i not in stopwords.words('english'):
                    x.append(i)
            return " ".join(x)

        def join_back (list_input):
            return " ".join(list_input)

        # Applying the corrected functions
        df['text'] = df['text'].apply(clean_html)
        df['title'] = df['title'].apply(clean_html)

        df['text'] = df['text'].apply(convert_lower)
        df['title'] = df['title'].apply(convert_lower)

        df['text'] = df['text'].apply(remove_special)
        df['title'] = df['title'].apply(remove_special)

        # Re-create 'content' and apply all preprocessing steps from scratch with corrected functions

        # Combine title and text
        df["content"] = df["title"] + " " + df["text"]

        # Apply cleaning, lowercasing, and special character removal
        df['content'] = df['content'].apply(clean_html)
        df['content'] = df['content'].apply(convert_lower)
        df['content'] = df['content'].apply(remove_special)

        # Apply stopword removal and join back into a string
        df['content']= df['content'].apply(remove_stopwords)

        # Display a sample of the cleaned content and its shape to confirm
        print(df['content'].head())
        print(df['content'].shape)

        # Redundant preprocessing steps, removed.

        remove_special('The %e @ Ckassic use of the word. it is called o2 as that is the nickname given to the as wald maximum security state')

        # Redundant preprocessing and nltk installation, removed.

        # Redundant and problematic function redefinition, removed. stopwords download moved to an earlier cell.

        # Redundant application of problematic remove_stopwords, removed.

        # Problematic stem_words function using global variable, removed. Stemming can be integrated into the main preprocessing if desired.

        # Demo of problematic stem_words, removed.

        # Redundant join_back definition and irrelevant x assignment, removed.

        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df["content"])

        df["label"] = df["result"]



        # X is already fit and transformed in cell xQHfHSJ4hWwH.
        # This cell should only be responsible for assigning y.
        y = df["label"]

        import pandas as pd
        from sklearn.model_selection import train_test_split

        # X is assumed to be correctly defined and have 100 samples from previous steps.
        # y is now directly taken from the processed df["label"]
        y = df["label"]

        # The train_test_split call now uses the correctly populated 'X' and 'y'.
        x_train , x_test , y_train, y_test = train_test_split(X , y , test_size = 0.2 , random_state = 2)
        x_train.shape

        x_test.shape

        y_train.shape

        y_test.shape

        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score
        model = MultinomialNB()
        model.fit(x_train , y_train)
        pred = model.predict(x_test)
        accuracy_score(y_test , pred)

        from sklearn.metrics import classification_report

        print(classification_report(y_test, pred))

        news = [news]

        news = tfidf.transform(news)

        prediction = model.predict(news)
        

        if prediction[0] == 1:
            output ="Real News"
        else:
            output = "Fake News"
    return render_template("homepage.html", output=output)

    import pickle

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        
        
if __name__ == '__main__':
    app.run(debug = True , port=5000)