import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

with open("intent_dataset.json") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

padded_sequences = pad_sequences(sequences, padding="post", maxlen=20)

label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)
categorical_labels = to_categorical(encoded_labels)

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(padded_sequences, categorical_labels, epochs=500, verbose=1)

final_train_acc = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc*100:.2f}%")
print(f"Final Training Loss: {final_train_loss:.4f}")

model.save("intent_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

