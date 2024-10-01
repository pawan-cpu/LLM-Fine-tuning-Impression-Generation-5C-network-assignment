import seaborn as sns
import pandas as pd

# Create a DataFrame for the top word pairs
top_word_pairs = pd.DataFrame({'Bigram': bigrams[:100], 'Frequency': X.sum(axis=0).tolist()[0][:100]})
sns.barplot(x='Frequency', y='Bigram', data=top_word_pairs)
plt.title("Top 100 Word Pairs")
plt.show()
