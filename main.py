import numpy as np
import tensorflow as tf


def recurrent_neural_network():
    # Загрузка текстового файла и предварительная обработка данных
    with open('V3001TH2.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_index = {char: index for index, char in enumerate(chars)}
    index_to_char = {index: char for index, char in enumerate(chars)}

    num_chars = len(chars)
    text_length = len(text)

    # Генерация тренировочных данных
    seq_length = 100  # Длина входной последовательности
    train_x = []
    train_y = []
    for i in range(0, text_length - seq_length, 1):
        input_seq = text[i:i + seq_length]
        output_seq = text[i + seq_length]
        train_x.append([char_to_index[char] for char in input_seq])
        train_y.append(char_to_index[output_seq])

    train_x = np.reshape(train_x, (len(train_x), seq_length, 1))
    train_x = train_x / float(num_chars)
    train_y = tf.keras.utils.to_categorical(train_y)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])),
        tf.keras.layers.Dense(num_chars, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Обучение модели
    model.fit(train_x, train_y, epochs=80, batch_size=128)

    # Генерация текста
    start_index = np.random.randint(0, len(train_x) - 1)
    start_seq = train_x[start_index]

    generated_text = ''
    for _ in range(500):
        x = np.reshape(start_seq, (1, len(start_seq), 1))
        x = x / float(num_chars)

        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = index_to_char[index]

        generated_text += result
        start_seq = np.append(start_seq, index)
        start_seq = start_seq[1:]

    with open('сгенерированный_текст.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)


if __name__ == '__main__':
    recurrent_neural_network()

