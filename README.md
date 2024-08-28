
# Лабораторная работа 3. Вариант 15

### Задание
Выполнить ранжирование признаков и решить с помощью библиотечной реализации дерева решений задачу классификации на 99% данных из курсовой работы. Проверить работу модели на оставшемся проценте, сделать вывод.

Модель:
- дерево решений DecisionTreeClassifier.

### Как запустить лабораторную работу
Для запуска программы необходимо с помощью командной строки в корневой директории файлов прокета прописать:
```
python main.py
```

### Какие технологии использовали
- Библиотека *numpy* для работы с массивами.
- Библиотека *pandas* для для работы с данными и таблицами.
- Библиотека *sklearn*:
    - *train_test_split* - для разделения данных на обучающую и тестовую выборки.
    - *MinMaxScaler* - для нормализации данных путем масштабирования значений признаков в диапазоне от 0 до 1.
    - *DecisionTreeClassifier* - для использования алгоритма дерева решений для задачи классификации.

### Описание лабораторной работы
#### Описание набора данных
В качестве набора данных был взят: *"Job Dataset"* - набор данных, содержащий объявления о вакансиях.
Набор данных состоит из следующих столбцов:
Descriptions for each of the columns in the dataset:
- Job Id - Уникальный идентификатор для каждой публикации вакансии.
- Experience - Требуемый или предпочтительный многолетний опыт работы на данной должности.
- Qualifications - Уровень образования, необходимый для работы.
- Salary Range - Диапазон окладов или компенсаций, предлагаемых за должность.
- Location - Город или область, где находится работа.
- Country - Страна, в которой находится работа.
- Latitude -  Координата широты местоположения работы.
- Longitude - Координата долготы местоположения работы.
- Work Type - Тип занятости (например, полный рабочий день, неполный рабочий день, контракт).
- Company Size - Приблизительный размер или масштаб компании, принимающей на работу.
- Job Posting Date - Дата, когда публикация о вакансии была опубликована.
- Preference - Особые предпочтения или требования к кандидатам (например, только мужчины или только женщины, или и то, и другое).
- Contact Person - Имя контактного лица или рекрутера для работы.
- Contact - Контактная информация для запросов о работе.
- Job Title - Название должности
- Role - Роль или категория работы (например, разработчик программного обеспечения, менеджер по маркетингу).
- Job Portal - Платформа или веб-сайт, на котором была размещена вакансия.
- Job Description - Подробное описание должностных обязанностей и требований.
- Benefits - Информация о льготах, предоставляемых в связи с работой (например, медицинская страховка, пенсионные планы).
- Skills - Навыки или квалификация, необходимые для работы.
- Responsibilities - Конкретные обязанности, связанные с работой.
- Company Name - Название компании, принимающей на работу.
- Company Profile - Краткий обзор истории компании и миссии.

Ссылка на страницу набора на kuggle: [Job Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

#### Подготовка данных
Для обеспечения качественного анализа данных и построения точных моделей машинного обучения, необходимо провести предварительную обработку данных. В данном проекте была выполнена следующая предобработка данных:
- Были удалены незначищие столбцы: *"Job Id", "latitude", "longitude", "Contact Person", "Contact", "Job Description", "Responsibilities"*.
```python
df_job.drop(["Job Id", "latitude", "longitude", "Contact Person", "Contact", "Job Description", "Responsibilities"], axis=1,
                inplace=True)
```
- Кодирование категориальных признаков, преобразованние их в уникальные числовые значения для каждого столбца, чтобы модель машинного обучения могла работать с ними, для столбцов: *'location', 'Country', 'Work Type','Preference', 'Job Title', 'Role', 'Job Portal', 'skills', 'Company',  'Sector'*. Пример кодирования категориальных признаков:
```python
# Создаем словарь для отображения уникальных значений в числовые идентификаторы
qualifications_dict = {qual: i for i, qual in enumerate(df_job['Qualifications'].unique())}
# Заменяем значения в столбце "Qualifications" соответствующими числовыми идентификаторами
df_job['Qualifications'] = df_job['Qualifications'].map(qualifications_dict)
```
- Данные столбцов *'Experience' и 'Salary Range'* были разделены соответственно на дополнительные столбцы: *'Min Experience', 'Max Experience', 'Min Salary', 'Max Salary'*. А сами столбцы *'Experience' и 'Salary Range'* удалены.
Пример разделения:
```python
# Разделяем значения 'Years' на минимальное и максимальное
# Удаляем символы валюты и другие символы
df_job['Experience'] = df_job['Experience'].apply(lambda x: str(x).replace('Years', '') if x is not None else x)
df_job[['Min Experience', 'Max Experience']] = df_job['Experience'].str.split(' to ', expand=True)
# Преобразуем значения в числовой формат
df_job['Min Experience'] = pd.to_numeric(df_job['Min Experience'])
df_job['Max Experience'] = pd.to_numeric(df_job['Max Experience'])
```
- Данные столбцы *'Job Posting Date'* были разбиты на дополнительные столбцы: *'year', 'month', 'day'*. А сам столбец *'Job Posting Date'* был удален.
- Данные ячеек столбца *'Company Profile'* имеют структуру вида *{"Sector":"Diversified","Industry":"Diversified Financials","City":"Sunny Isles Beach","State":"Florida","Zip":"33160","Website":"www.ielp.com","Ticker":"IEP","CEO":"David Willetts"}*, поэтому были разделены на дополнительные столбцы и закодированы для избежания категориальных признаков: *'Sector', 'Industry', 'City', 'State', 'Ticker'*, а данные о *'Zip', 'Website', 'CEO'* были удалены, как наименее важные. Также был удален сам столбец *'Company Profile'*.

#### Выявление значимых параметров

Создаем переменную y, которая содержит значения целевой переменной *"Qualifications"* из нашего подготовленного набора данных `data`. Разделяем данные на обучающую и тестовую выборки, где `corr.values` содержит значения признаков, которые будут использоваться для обучения модели, `y.values` содержит значения целевой переменной, а `test_size=0.2` указывает, что 20% данных будет использоваться для тестирования модели. Затем создаем экземпляр классификатора `DecisionTreeClassifier` и обучаем классификатор на обучающих данных. После чего получаем важности признаков из обученной модели, которые показывают, насколько сильно каждый признак влияет на прогнозы модели.
```python
 # определение целевой переменной
y = data['Qualifications']
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(corr.values, y.values, test_size=0.2)
# Создание экземпляра классификатора дерева решений
clf = DecisionTreeClassifier(random_state=241)
# Обучение модели на обучающей выборке
clf.fit(X_train, y_train)
# Прогнозирование классов для тестовой выборки
y_pred = clf.predict(X_test)
importances = clf.feature_importances_
print("Важность признаков: ")
print(importances)
print("Отсортированная важность признаков: ")
conversion_ratings(importances)
```
Для того, чтобы получить отсортированный список важности признаков и их значения создаем дополнительный метод `conversion_ratings` с аналогичной логикой работы сортировки данных, как в лабораторной работе 2.

После запуска имеем следующий результат:
```
Важность признаков: 
[0.04535517 0.04576875 0.03236705 0.07819966 0.02279837 0.0608208
 0.04189454 0.04985896 0.0418959  0.03571376 0.03675038 0.04229454
 0.04054691 0.05188657 0.03849015 0.04226668 0.04105321 0.03616932
 0.03535738 0.01584379 0.04569225 0.0588709  0.00620841 0.00620682
 0.00606359 0.00595985 0.00568906 0.00345068 0.00343211 0.00491702
 0.00614867 0.00568446 0.00634429]
Отсортированная важность признаков: 
{'Company Size': 1.0, 'Job Title': 0.77, 'day': 0.74, 'Max Salary': 0.65, 'Job Portal': 0.62, 'Country': 0.57, 'month': 0.57, 'location': 0.56, 'Max Experience': 0.52, 'Industry': 0.52, 'Role': 0.51, 'skills': 0.51, 'Min Salary': 0.5, 'City': 0.5, 'Sector': 0.47, 'Min Experience': 0.45, 'State': 0.44, 'Company': 0.43, 'Ticker': 0.43, 'Work Type': 0.39, 'Preference': 0.26, 'year': 0.17, "'Casual Dress Code, Social and Recreational Activities, Employee Referral Programs, Health and Wellness Facilities, Life and Disability Insurance'": 0.04, "'Childcare Assistance, Paid Time Off (PTO), Relocation Assistance, Flexible Work Arrangements, Professional Development'": 0.04, "'Employee Assistance Programs (EAP), Tuition Reimbursement, Profit-Sharing, Transportation Benefits, Parental Leave'": 0.04, "'Life and Disability Insurance, Stock Options or Equity Grants, Employee Recognition Programs, Health Insurance, Social and Recreational Activities'": 0.04, "'Tuition Reimbursement, Stock Options or Equity Grants, Parental Leave, Wellness Programs, Childcare Assistance'": 0.04, "'Employee Referral Programs, Financial Counseling, Health and Wellness Facilities, Casual Dress Code, Flexible Spending Accounts (FSAs)'": 0.03, "'Flexible Spending Accounts (FSAs), Relocation Assistance, Legal Assistance, Employee Recognition Programs, Financial Counseling'": 0.03, "'Transportation Benefits, Professional Development, Bonuses and Incentive Programs, Profit-Sharing, Employee Discounts'": 0.03, "'Legal Assistance, Bonuses and Incentive Programs, Wellness Programs, Employee Discounts, Retirement Plans'": 0.02, "'Health Insurance, Retirement Plans, Flexible Work Arrangements, Employee Assistance Programs (EAP), Bonuses and Incentive Programs'": 0.0, "'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'": 0.0}

```

### Вывод

Таким образом, можно сделать вывод о том, что наиболее важным признаком является "Company Size" с важностью 1.0, за ним следуют "Job Title" (0.77), "day" (0.74) и "Max Salary" (0.65). Исходя из значений важности признаков, можно сделать вывод, что как числовые, так и категориальные признаки вносят вклад в прогнозирование целевой переменной. 

В целом, результаты лабораторной работы позволяют оценить важность каждого признака в прогнозировании целевой переменной и помогают понять, какие признаки следует учитывать при анализе данных и принятии решений.