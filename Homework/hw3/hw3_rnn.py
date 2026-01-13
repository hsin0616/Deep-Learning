import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers

data_URL = 'shakespeare_train.txt'
data_URL2 = 'shakespeare_valid.txt'
with io.open(data_URL, 'r', encoding='utf8') as f:
    text = f.read()
with io.open(data_URL2, 'r', encoding='utf8') as f2:
    text2 = f2.read()

# create vocab and index 
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = np.array(vocab)

train_data = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
valid_data = np.array([vocab_to_int[c] for c in text2], dtype=np.int32)

input_length = 150

# create tf.data Dataset
new_dataset = tf.data.Dataset.from_tensor_slices(train_data)
sequences = new_dataset.batch(input_length + 1, drop_remainder=True)

new_valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
vsequences = new_valid_dataset.batch(input_length + 1, drop_remainder=True)

def input_and_labels(sentence):
    input_text = sentence[:-1]
    target_text = sentence[1:]
    return input_text, target_text

dataset = sequences.map(input_and_labels)
vdataset = vsequences.map(input_and_labels)

# batch / buffer
bsize = 128
buf = 10000
dataset = dataset.shuffle(buf).batch(bsize, drop_remainder=True)
vdataset = vdataset.batch(bsize, drop_remainder=True)   # validation 不一定要 shuffle

vocab_len = len(vocab)
embdim = 64
unit = 128
batch_size = 128

# create standard RNN model（SimpleRNN）
model = tf.keras.Sequential()
model.add(
    layers.Embedding(
        input_dim=vocab_len,
        output_dim=embdim,
        batch_input_shape=[batch_size, None]
    )
)

model.add(
    layers.SimpleRNN(
        unit,
        activation='tanh',
        return_sequences=True,
        stateful=True
    )
)

model.add(layers.Dense(vocab_len))
model.summary()

for inputseq, outseq in dataset.take(1):
    preseq = model(inputseq)

def cost(output, predict):
    return tf.keras.losses.sparse_categorical_crossentropy(
        output, predict, from_logits=True
    )

model.compile(optimizer='adam', loss=cost)

EPOCHS = 10

history = model.fit(
    dataset,
    epochs=EPOCHS,
    validation_data=vdataset
)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

print("Final Training error rate = ", loss_values[-1])
print("Final Validation error rate = ", val_loss_values[-1])

epochs = range(1, len(loss_values) + 1)

plt.figure(1)
plt.plot(epochs, loss_values, label='Training error rate')
plt.plot(epochs, val_loss_values, label='Validation error rate')
plt.title('Training and Validation Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error (Loss)')
plt.legend()
plt.grid(True)
plt.show()
