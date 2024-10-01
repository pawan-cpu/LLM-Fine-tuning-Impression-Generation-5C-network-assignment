import plotly.express as px

fig = px.bar(top_word_pairs, x='Frequency', y='Bigram', title='Top 100 Word Pairs')
fig.show()
