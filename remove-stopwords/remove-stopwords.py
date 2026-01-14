def remove_stopwords(tokens, stopwords):
    return [k for k in tokens if k not in stopwords]
