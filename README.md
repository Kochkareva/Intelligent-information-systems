
# Лабораторная работа 7. Вариант 15

### Задание
Выбрать художественный текст (нечетные варианты – англоязычный) и обучить на нем рекуррентную нейронную сеть
для решения задачи генерации. Подобрать архитектуру и параметры так,
чтобы приблизиться к максимально осмысленному результату.


### Как запустить лабораторную работу
Для запуска программы необходимо с помощью командной строки в корневой директории файлов прокета прописать:
```
python main.py
```
### Какие технологии использовали
- Библиотека *numpy* для работы с массивами.
- Библиотека *tensorflow* - для машинного обучения. Она предоставляет инструменты для создания и обучения различных моделей машинного обучения, включая нейронные сети.

### Описание лабораторной работы
Для данной лабораторной работы был взят текст на 1596 строк текста.

```python
 with open('V3001TH2.txt', 'r', encoding='utf-8') as f:
        text = f.read()
```

Далее создали  список уникальных символов `chars`, а также словари `char_to_index` и `index_to_char`, которые используются для преобразования символов в индексы и наоборот.

```python
chars = sorted(list(set(text)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}
```

После чего можем генерировать ренировочные данные `train_x` и `train_y`. `train_x` содержит последовательности символов длиной `seq_length` из текста, а `train_y` содержит следующий символ после каждой входной последовательности. Каждый символ преобразуется в соответствующий индекс, используя словарь `char_to_index`.

```python
# Генерация тренировочных данных
seq_length = 100  # Длина входной последовательности
train_x = []
train_y = []
for i in range(0, text_length - seq_length, 1):
    input_seq = text[i:i + seq_length]
    output_seq = text[i + seq_length]
    train_x.append([char_to_index[char] for char in input_seq])
    train_y.append(char_to_index[output_seq])
```

Далее преобразуем `train_x` в трехмерный массив с размерностью (количество примеров, `seq_length`, 1).
Нормализуем значения `train_x` путем деления на `num_chars` и преобразуем `train_y` в `one-hot` представление с помощью `tf.keras.utils.to_categorical.`

```python
train_x = np.reshape(train_x, (len(train_x), seq_length, 1))
train_x = train_x / float(num_chars)
train_y = tf.keras.utils.to_categorical(train_y)
```
Теперь переходим к созданию модели рекуррентной нейронной сети с `LSTM` слоем, принимающим входные данные размерности `(train_x.shape[1], train_x.shape[2])` и плотным слоем с активацией softmax.
Компилируем модель с функцией потерь `categorical_crossentropy` и оптимизатором `adam`.

```python
model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])),
        tf.keras.layers.Dense(num_chars, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
```

Обучаем модель на тренировачных данных с заданным количеством эпох - 80 и размером пакета - 128.
```
model.fit(train_x, train_y, epochs=80, batch_size=128)
```

Генерируется текст, начиная с случайного индекса `start_index` в `train_x`. Затем, на каждой итерации цикла, модель предсказывает следующий символ, добавляет его к сгенерированному тексту и обновляет `start_seq` для использования в следующей итерации.
Записывает сгенерированный текст в файл *'сгенерированный_текст.txt'*.

Результат выполнения:

```
Ih ses shven they to tore a fit oo th toie th sook a buck and tore tote a siee fot oo the searen.
Jnd buonds sore toee th the shele and thans to the siee and soans tie his and tooning tie hit cnd toens the his and croninng his bioter. 
 

— Iod you ducking tooeeds so toieg a buck and to bor aeeut tore a sigee oo toire a ducn fo toine to see sooeee oo the saelen. Tnd blond toees the sirt and that the sooel and thai to the soeee of the shale. 
 

"Iotk toe ffcrtes," Vincent says suth a suine and a
```
### Вывод

Текст содержит некоторые слова и фразы, которые кажутся некорректными или непонятными. Это может быть связано с недостаточным количеством обучающих данных или эпох обучения.