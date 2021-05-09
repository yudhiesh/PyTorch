from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

corpus = ["Time flies flies like an arrow.", "Fruit flies like a banana."]
corpusData = [word.split(" ") for word in corpus]
corpusData_ = [w for word in corpusData for w in word]
corpusData_ = list(map(lambda x: x.lower(), corpusData_))
vocab = list(map(lambda x: x.rstrip("."), corpusData_))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(
    tfidf,
    annot=True,
    cbar=False,
    xticklabels=vocab,
    yticklabels=["Sentence 1", "Sentence 2"],
)
