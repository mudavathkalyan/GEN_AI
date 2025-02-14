##Natural Language Processing (NLP) Overview
##NLP Workflow:
📄 Input Text → ✂️ Tokenization → 🛑 Stopwords Removal → 🔄 Stemming/Lemmatization → ✨ Cleaned Text → 🔢 Vectorization → 🧠 Model Training

🔹 1. Tokenization
Breaking text into words, sentences, or subwords (tokens).

🔹 Types:

Word Tokenization: Splits text into words.
Sentence Tokenization: Splits text into sentences.
Subword Tokenization: Useful for languages with complex words.
📌 Example (Using NLTK & spaCy):

python
Copy
Edit
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLP is amazing! It's transforming AI."
print(word_tokenize(text))  # ['NLP', 'is', 'amazing', '!', "It", "'s", 'transforming', 'AI', '.']

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)  # ['NLP', 'is', 'amazing', '!', 'It', "'s", 'transforming', 'AI', '.']
🔹 2. Text Preprocessing
Cleaning raw text for better model performance.

🔹 Steps:

Lowercasing (Convert text to lowercase)
Removing Punctuation & Special Characters
Removing Numbers
Removing Extra Whitespaces
📌 Example:

python
Copy
Edit
import re

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

text = "Hello! NLP is evolving in 2025 🚀."
print(preprocess(text))  # "hello nlp is evolving in"
🔹 3. Stopwords Removal
Stopwords are common words (e.g., "the", "is", "and") that don't add meaning.

📌 Example (Using NLTK & spaCy):

python
Copy
Edit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))
text = "NLP is an important field of artificial intelligence."
words = word_tokenize(text)
filtered_text = [word for word in words if word.lower() not in stop_words]
print(filtered_text)  # ['NLP', 'important', 'field', 'artificial', 'intelligence']
🔹 4. Stemming & Lemmatization
Reduce words to their root/base forms.

🔹 Stemming: Removes suffixes (e.g., "running" → "run").
🔹 Lemmatization: Converts words to dictionary form (e.g., "better" → "good").

📌 Example (Using NLTK & spaCy):

python
Copy
Edit
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
print(stemmer.stem(word))  # 'run'
print(lemmatizer.lemmatize(word, pos="v"))  # 'run'
🔹 5. Vectorization (Converting Text to Numbers)
Convert text into numerical format for machine learning models.

🔹 Techniques:

Bag of Words (BoW)
TF-IDF (Term Frequency-Inverse Document Frequency)
Word Embeddings (Word2Vec, GloVe, FastText)
📌 Example (Using Scikit-learn TF-IDF):

python
Copy
Edit
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["NLP is fun", "Machine learning is great"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())  # ['fun', 'great', 'is', 'learning', 'machine', 'nlp']
print(X.toarray())  # TF-IDF values
🔹 6. Word Embeddings (Word2Vec, FastText)
Capture word meanings based on context.

🔹 Word2Vec (Trained on context)
🔹 FastText (Handles subwords, better for rare words)

📌 Example (Using Gensim Word2Vec):

python
Copy
Edit
from gensim.models import Word2Vec

sentences = [["NLP", "is", "amazing"], ["Deep", "learning", "is", "powerful"]]
model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)

print(model.wv.most_similar("NLP"))  # Finds similar words
🔹 7. NLP Model Training (Naïve Bayes Example)
Train a model to classify text (e.g., Spam Detection).

📌 Example (Using Scikit-learn):

python
Copy
Edit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

X_train = ["This is a great product", "Free money now!", "Discount offer available"]
y_train = ["ham", "spam", "spam"]

clf.fit(X_train, y_train)

# Test
print(clf.predict(["Get a 50% discount"]))  # Output: ['spam']
🔹 8. Topic Modeling (LDA - Latent Dirichlet Allocation)
Discover hidden topics in text.

📌 Example (Using Gensim LDA):

python
Copy
Edit
from gensim import corpora, models

texts = [["NLP", "is", "amazing"], ["Deep", "learning", "is", "powerful"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
print(lda.print_topics())

