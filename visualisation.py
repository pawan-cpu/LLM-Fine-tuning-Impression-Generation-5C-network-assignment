from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Tokenize and get the top word pairs (bigrams)
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(corpus)
bigrams = vectorizer.get_feature_names_out()

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(bigrams))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
