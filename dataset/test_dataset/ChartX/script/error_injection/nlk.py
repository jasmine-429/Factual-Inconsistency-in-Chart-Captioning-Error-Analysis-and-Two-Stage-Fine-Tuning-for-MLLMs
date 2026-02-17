import nltk

print("NLTK data paths:")
for path in nltk.data.path:
    print(" -", path)
