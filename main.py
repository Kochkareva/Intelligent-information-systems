import os.path
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Задачи анализа, решаемые регрессией: предсказать диапазон окладов или компенсаций
# на основе других признаков: опыт, квалификация, тип работы и т.д.


picfld = os.path.join('static', 'charts')

data = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/updated_job_descriptions.csv')

def linear_regression_min_salary():
    y = data['Min Salary']
    df = data.copy()
    # удаляем целевое значение и наименее важные параметры
    df.drop(['Min Salary', 'Preference', 'day', 'month', 'Job Portal', 'State', 'location', 'Country', 'Industry', 'skills',
             'Role', 'Job Title', 'City', 'Sector', 'Ticker', 'Company Size', 'Company', 'Max Salary'],
            axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0002, train_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r_sq = model.score(X_test, y_test)
    plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    plt.plot(y_pred, c="#00BFFF",
             label="\"y\" предсказанная \n" "Кд = " + str(r_sq))
    plt.legend(loc='lower left')
    plt.title("Линейная регрессия")
    plt.savefig('static/charts/MinSalaryChart.png')
    plt.close()


def linear_regression_max_salary():
    y = data['Max Salary']
    df = data.copy()
    # удаляем целевое значение и наименее важные параметры
    df.drop(['Max Salary',  'Max Experience', 'Job Portal', 'day', 'Sector', 'Industry', 'Country', 'location', 'Job Title',
             'Ticker', 'Company', 'City', 'State', 'Role', 'skills', 'Qualifications', 'Company Size', 'Min Salary'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0002, train_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r_sq = model.score(X_test, y_test)
    plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    plt.plot(y_pred, c="#00BFFF",
             label="\"y\" предсказанная \n" "Кд = " + str(r_sq))
    plt.legend(loc='lower left')
    plt.title("Линейная регрессия")
    plt.savefig('static/charts/MaxSalaryChart.png')
    plt.close()


# оценка важности параметров
def RFE_max_salary():
    df = data.copy()
    y = data['Max Salary']
    df.drop(["Max Salary"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.2)
    column_names = ['Qualifications', 'Country', 'location', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
                    'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
                    'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day',
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
    estimator = LinearRegression()
    rfe_model = RFE(estimator)
    rfe_model.fit(X_train, y_train)
    ranks = rank_to_dict_rfe(rfe_model.ranking_, column_names)
    sorted_dict = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))
    print("Оценка важности параметров для нахождения максимальной оплаты труда")
    print(sorted_dict)


def RFE_min_salary():
    df = data.copy()
    y = data['Min Salary']
    df.drop(["Min Salary"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.2)
    column_names = ['Qualifications', 'Country', 'location', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
                    'skills', 'Company', 'Min Experience', 'Max Experience', 'Max Salary',
                    'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day',
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
    estimator = LinearRegression()
    rfe_model = RFE(estimator)
    rfe_model.fit(X_train, y_train)
    ranks = rank_to_dict_rfe(rfe_model.ranking_, column_names)
    sorted_dict = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))
    print("Оценка важности параметров для нахождения минимальной оплаты труда")
    print(sorted_dict)


def rank_to_dict_rfe(ranking, names):
    n_ranks = [float(1 / i) for i in ranking]
    n_ranks = map(lambda x: round(x, 2), n_ranks)
    return dict(zip(names, n_ranks))


if __name__ == '__main__':
    # linear_regression_min_salary()
    # linear_regression_max_salary()
    RFE_min_salary()
    RFE_max_salary()

