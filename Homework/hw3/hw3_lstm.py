import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers

# ===================== 讀資料 =====================
data_URL = 'shakespeare_train.txt'
data_URL2 = 'shakespeare_valid.txt'

with io.open(data_URL, 'r', encoding='utf8') as f:
    text = f.read()
with io.open(data_URL2, 'r', encoding='utf8') as f2:
    text2 = f2.read()

# create vocabulary
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = np.array(vocab)

# text to index
train_data = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
valid_data = np.array([vocab_to_int.get(c, 0) for c in text2], dtype=np.int32)
#  vocab_to_int.get(c, 0), avoiding errors occur when unseen character exists

# ===================== Dataset =====================
input_length = 150

# training sequences
new_dataset = tf.data.Dataset.from_tensor_slices(train_data)
sequences = new_dataset.batch(input_length + 1, drop_remainder=True)

# validation sequences
new_valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
vsequences = new_valid_dataset.batch(input_length + 1, drop_remainder=True)

def input_and_labels(sentence):
    input_text = sentence[:-1]
    target_text = sentence[1:]
    return input_text, target_text

dataset = sequences.map(input_and_labels)
vdataset = vsequences.map(input_and_labels)

batch_size = 128
buf = 10000

dataset = dataset.shuffle(buf).batch(batch_size, drop_remainder=True)
vdataset = vdataset.batch(batch_size, drop_remainder=True)

# ===================== create the model =====================
vocab_len = len(vocab)
embdim = 64
unit = 128

model = tf.keras.Sequential()
model.add(layers.Embedding(
    input_dim=vocab_len,
    output_dim=embdim,
    batch_input_shape=[batch_size, None]
))
# Stateful LSTM
model.add(layers.LSTM(
    unit,
    activation='tanh',
    return_sequences=True,
    stateful=True
))
model.add(layers.Dense(vocab_len))

model.summary()

# ===================== loss & compile =====================
def cost(y_true, y_pred):
    # y_true: (batch, time) 的 index
    # y_pred: (batch, time, vocab_len) 的 logits
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )

model.compile(optimizer='adam', loss=cost)

# ===================== training (include validation) =====================
train_history = model.fit(
    dataset,
    epochs=10,
    validation_data=vdataset
)

# ===================== read loss & plot curve =====================
history_dict = train_history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

print("Final Training error rate = ", loss_values[-1])
print("Final Validation error rate = ", val_loss_values[-1])

epochs = range(1, len(loss_values) + 1)

plt.figure()
plt.plot(epochs, loss_values, label='Training loss')
plt.plot(epochs, val_loss_values, label='Validation loss')
# plt.plot(epochs, loss_values, marker='o', label='Training loss')
# plt.plot(epochs, val_loss_values, marker='o', label='Validation loss')
plt.title('Training and Validation Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ===== 5. =====
# create a model with batch_size=1 to generate words
gen_model = tf.keras.Sequential([
    layers.Embedding(vocab_len, embdim, batch_input_shape=[1, None]),
    layers.LSTM(unit, return_sequences=True, stateful=True),
    layers.Dense(vocab_len)
])


gen_model.set_weights(model.get_weights())

def generate_text(model, start_string, num_generate=300, temperature=1.0):
    input_eval = [vocab_to_int[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)  # shape: (1, seq_len)

    text_generated = []

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)          # (1, seq_len, vocab_len)
        predictions = predictions[:, -1, :]      # get the last output
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)     # next input
        text_generated.append(int_to_vocab[predicted_id])  # put it into the array

    return start_string + ''.join(text_generated)


# 1: starts from ROMIO
text1 = generate_text(gen_model, start_string="ROMEO: ", num_generate=300, temperature=0.5)
print(text1)

# 2: starts from JULIET
text2 = generate_text(gen_model, start_string="JULIET: ", num_generate=300, temperature=0.5)
print(text2)

