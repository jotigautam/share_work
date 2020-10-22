import nltk
nltk.download('punkt')

paragraph = "In order to excel in a chaotic, competitive environment your team needs to play at world-class levels like the teams at organizations such as FedEx, GE, IBM, Castrol, Unilever, Microsoft, Yale University and NASA. Through his professional development training Robin Sharma has helped these and many other enterprises unleash the highest potential of their people and develop employees who Lead Without a Title so that your company reaches its strategic objectives fast."

sentences = nltk.sent_tokenize(paragraph)

words = nltk.word_tokenize(paragraph)

