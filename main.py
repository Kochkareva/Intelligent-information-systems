import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def data_preprocessing():
    df_job_orig = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/job_descriptions.csv')
    df_job_orig = pd.DataFrame(df_job_orig)
    desired_rows = int(0.99 * len(df_job_orig))
    df_job = df_job_orig.copy()
    df_job = df_job[:desired_rows]
    df_job.drop(["Job Id", "latitude", "longitude", "Contact Person", "Contact", "Job Description", "Responsibilities"], axis=1,
                inplace=True)
    # digitization
    # --------------------------'Years'------------------------
    # Разделяем значения 'Years' на минимальное и максимальное
    # Удаляем символы валюты и другие символы
    df_job['Experience'] = df_job['Experience'].apply(lambda x: str(x).replace('Years', '') if x is not None else x)
    df_job[['Min Experience', 'Max Experience']] = df_job['Experience'].str.split(' to ', expand=True)
    # Преобразуем значения в числовой формат
    df_job['Min Experience'] = pd.to_numeric(df_job['Min Experience'])
    df_job['Max Experience'] = pd.to_numeric(df_job['Max Experience'])
    # --------------------------'Salary Range'------------------------
    # Удаляем символы валюты и другие символы
    df_job['Salary Range'] = df_job['Salary Range'].str.replace('$', '').str.replace('K', '000')
    # Разделяем значения на минимальное и максимальное
    df_job[['Min Salary', 'Max Salary']] = df_job['Salary Range'].str.split('-', expand=True)
    # Преобразуем значения в числовой формат
    df_job['Min Salary'] = pd.to_numeric(df_job['Min Salary'])
    df_job['Max Salary'] = pd.to_numeric(df_job['Max Salary'])
    # --------------------------'Qualifications'------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    qualifications_dict = {qual: i for i, qual in enumerate(df_job['Qualifications'].unique())}
    # Заменяем значения в столбце "Qualifications" соответствующими числовыми идентификаторами
    df_job['Qualifications'] = df_job['Qualifications'].map(qualifications_dict)
    # --------------------------'location'------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    locations_dict = {locat: i for i, locat in enumerate(df_job['location'].unique())}
    # Заменяем значения в столбце "location" соответствующими числовыми идентификаторами
    df_job['location'] = df_job['location'].map(locations_dict)
    # --------------------------'Country'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    countries_dict = {countr: i for i, countr in enumerate(df_job['Country'].unique())}
    # Заменяем значения в столбце "Country" соответствующими числовыми идентификаторами
    df_job['Country'] = df_job['Country'].map(countries_dict)
    # --------------------------'Work Type'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    wt_dict = {wt: i for i, wt in enumerate(df_job['Work Type'].unique())}
    # Заменяем значения в столбце "Work Type" соответствующими числовыми идентификаторами
    df_job['Work Type'] = df_job['Work Type'].map(wt_dict)
    # --------------------------'Preference gender'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    gender_dict = {gender: i for i, gender in enumerate(df_job['Preference'].unique())}
    # Заменяем значения в столбце "Preference" соответствующими числовыми идентификаторами
    df_job['Preference'] = df_job['Preference'].map(gender_dict)
    # --------------------------'Job Title'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    jt_dict = {jt: i for i, jt in enumerate(df_job['Job Title'].unique())}
    # Заменяем значения в столбце "Job Title" соответствующими числовыми идентификаторами
    df_job['Job Title'] = df_job['Job Title'].map(jt_dict)
    # --------------------------'Role'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    role_dict = {role: i for i, role in enumerate(df_job['Role'].unique())}
    # Заменяем значения в столбце "Role" соответствующими числовыми идентификаторами
    df_job['Role'] = df_job['Role'].map(role_dict)
    # --------------------------'Job Portal'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    jp_dict = {jp: i for i, jp in enumerate(df_job['Job Portal'].unique())}
    # Заменяем значения в столбце "Job Portal" соответствующими числовыми идентификаторами
    df_job['Job Portal'] = df_job['Job Portal'].map(jp_dict)
    # --------------------------'Company'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {comp: i for i, comp in enumerate(df_job['Company'].unique())}
    # Заменяем значения в столбце "Company" соответствующими числовыми идентификаторами
    df_job['Company'] = df_job['Company'].map(comp_dict)
    # --------------------------'Company Profile'-------------------------
    df_company_profile = df_job['Company Profile'].str.split('",', expand=True)
    df_company_profile.columns = ['Sector', 'Industry', 'City', 'State', 'Zip', 'Website', 'Ticker', 'CEO']
    df_company_profile = df_company_profile.apply(
        lambda x: x.str.replace('{', '').str.replace('"', '').str.replace('}', '')
        .str.replace('Sector', '').str.replace('Industry', '').str.replace('City', '')
        .str.replace('State', '').str.replace('Zip', '').str.replace('Website', '')
        .str.replace('Ticker', '').str.replace('CEO', '').str.replace(':', ''))
    df_company_profile.drop(["CEO", "Website", "Zip"], axis=1, inplace=True)
        # --------------------------'Sector'-------------------------
        # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {sector: i for i, sector in enumerate(df_company_profile['Sector'].unique())}
        # Заменяем значения в столбце "Sector" соответствующими числовыми идентификаторами
    df_company_profile['Sector'] = df_company_profile['Sector'].map(comp_dict)
        # --------------------------'Industry'-------------------------
        # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {industry: i for i, industry in enumerate(df_company_profile['Industry'].unique())}
        # Заменяем значения в столбце "Industry" соответствующими числовыми идентификаторами
    df_company_profile['Industry'] = df_company_profile['Industry'].map(comp_dict)
        # --------------------------'City'-------------------------
        # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {city: i for i, city in enumerate(df_company_profile['City'].unique())}
        # Заменяем значения в столбце "City" соответствующими числовыми идентификаторами
    df_company_profile['City'] = df_company_profile['City'].map(comp_dict)
        # --------------------------'State'-------------------------
        # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {state: i for i, state in enumerate(df_company_profile['State'].unique())}
        # Заменяем значения в столбце "State" соответствующими числовыми идентификаторами
    df_company_profile['State'] = df_company_profile['State'].map(comp_dict)
        # --------------------------'Ticker'-------------------------
        # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {ticker: i for i, ticker in enumerate(df_company_profile['Ticker'].unique())}
        # Заменяем значения в столбце "Ticker" соответствующими числовыми идентификаторами
    df_company_profile['Ticker'] = df_company_profile['Ticker'].map(comp_dict)
    # Объединение преобразованных столбцов с исходным датасетом
    df_job = pd.concat([df_job, df_company_profile], axis=1)
    # --------------------------'Job Posting Date'-------------------------
    df_job[['year', 'month', 'day']] = df_job['Job Posting Date'].str.split('-', expand=True)
    df_job['year'] = pd.to_numeric(df_job['year'])
    df_job['month'] = pd.to_numeric(df_job['month'])
    df_job['day'] = pd.to_numeric(df_job['day'])
    # --------------------------'Benefits'-------------------------
    df_job['Benefits'] = df_job['Benefits'].str.replace('{', '').str.replace('}', '')
    # Применить метод get_dummies для оцифровки столбца 'Benefits'
    benefits_encoded = pd.get_dummies(df_job['Benefits'], dtype=int)
    # Соединить исходный DataFrame с оцифрованными данными
    df_job = pd.concat([df_job, benefits_encoded], axis=1)
    # --------------------------'skills'-------------------------
    # Создаем словарь для отображения уникальных значений в числовые идентификаторы
    comp_dict = {skill: i for i, skill in enumerate(df_job['skills'].unique())}
    # Заменяем значения в столбце "skills" соответствующими числовыми идентификаторами
    df_job['skills'] = df_job['skills'].map(comp_dict)
    df_job.drop(["Company Profile", "Experience", "Salary Range", "Benefits", "Job Posting Date"], axis=1, inplace=True)
    print(df_job.dtypes)
    df_job.to_csv('D:/Интеллектуальные информационные системы/Dataset/updated_job_descriptions.csv', index=False)


def decision_tree_classifier():
    data = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/updated_job_descriptions.csv')
    corr = data[['location', 'Country', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal', 'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
                 'Max Salary', 'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day',
                 "'Casual Dress Code, Social and Recreational Activities, Employee Referral Programs, Health and Wellness Facilities, Life and Disability Insurance'",
                 "'Childcare Assistance, Paid Time Off (PTO), Relocation Assistance, Flexible Work Arrangements, Professional Development'",
                 "'Employee Assistance Programs (EAP), Tuition Reimbursement, Profit-Sharing, Transportation Benefits, Parental Leave'",
                 "'Employee Referral Programs, Financial Counseling, Health and Wellness Facilities, Casual Dress Code, Flexible Spending Accounts (FSAs)'",
                 "'Flexible Spending Accounts (FSAs), Relocation Assistance, Legal Assistance, Employee Recognition Programs, Financial Counseling'",
                 "'Health Insurance, Retirement Plans, Flexible Work Arrangements, Employee Assistance Programs (EAP), Bonuses and Incentive Programs'",
                 "'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'",
                 "'Legal Assistance, Bonuses and Incentive Programs, Wellness Programs, Employee Discounts, Retirement Plans'",
                 "'Life and Disability Insurance, Stock Options or Equity Grants, Employee Recognition Programs, Health Insurance, Social and Recreational Activities'",
                 "'Transportation Benefits, Professional Development, Bonuses and Incentive Programs, Profit-Sharing, Employee Discounts'",
                 "'Tuition Reimbursement, Stock Options or Equity Grants, Parental Leave, Wellness Programs, Childcare Assistance'"]]
    print(corr.head())
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


def conversion_ratings(rank):
    column_names = ['location', 'Country', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal', 'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
                 'Max Salary', 'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day',
                 "'Casual Dress Code, Social and Recreational Activities, Employee Referral Programs, Health and Wellness Facilities, Life and Disability Insurance'",
                 "'Childcare Assistance, Paid Time Off (PTO), Relocation Assistance, Flexible Work Arrangements, Professional Development'",
                 "'Employee Assistance Programs (EAP), Tuition Reimbursement, Profit-Sharing, Transportation Benefits, Parental Leave'",
                 "'Employee Referral Programs, Financial Counseling, Health and Wellness Facilities, Casual Dress Code, Flexible Spending Accounts (FSAs)'",
                 "'Flexible Spending Accounts (FSAs), Relocation Assistance, Legal Assistance, Employee Recognition Programs, Financial Counseling'",
                 "'Health Insurance, Retirement Plans, Flexible Work Arrangements, Employee Assistance Programs (EAP), Bonuses and Incentive Programs'",
                 "'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'",
                 "'Legal Assistance, Bonuses and Incentive Programs, Wellness Programs, Employee Discounts, Retirement Plans'",
                 "'Life and Disability Insurance, Stock Options or Equity Grants, Employee Recognition Programs, Health Insurance, Social and Recreational Activities'",
                 "'Transportation Benefits, Professional Development, Bonuses and Incentive Programs, Profit-Sharing, Employee Discounts'",
                 "'Tuition Reimbursement, Stock Options or Equity Grants, Parental Leave, Wellness Programs, Childcare Assistance'"]
    ranks = dict()
    ranks = np.abs(rank)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(33, 1)).ravel()  # - преобразование данных
    ranks = map(lambda x: round(x, 2), ranks)  # - округление элементов массива
    my_dict = dict(zip(column_names, ranks))
    sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1], reverse=True))
    print(sorted_dict)


if __name__ == '__main__':
    #data_preprocessing()
    decision_tree_classifier()