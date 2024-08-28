import os.path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


picfld = os.path.join('static', 'charts')

data = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/updated_job_descriptions.csv')
data_orig = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/job_descriptions.csv')
y = data['Country']


def k_means():
    df = data.copy()
    df.drop(['Country', 'location', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
             'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
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
             "'Tuition Reimbursement, Stock Options or Equity Grants, Parental Leave, Wellness Programs, Childcare Assistance'"],
            axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(X_train.values)
    labels = kmeans.predict(X_test.values)
    centroids = kmeans.cluster_centers_
    print("Координаты центроидов:", centroids)
    plt.scatter(X_test['Qualifications'], X_test['Work Type'], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red')
    plt.xlabel('Qualifications')
    plt.ylabel('Work Type')
    plt.title('KMeans Clustering')
    plt.savefig('static/charts/KMeansClustering.png')
    plt.close()
    print("Уникальных Work Type :", data['Work Type'].nunique())
    print("Уникальных Qualifications:", data['Qualifications'].nunique())
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        y_values = data_orig.loc[indices, 'Country'].values
        print(f"Значения y для кластера {label}: {y_values}")
    # Оценка силуэтного коэффициента
    silhouette = silhouette_score(X_test.values, kmeans.predict(X_test.values))
    print("Силуэтный коэффициент:", silhouette)
    # Оценка индекса Дэвиса-Болдина
    davies_bouldin = davies_bouldin_score(X_test.values, kmeans.predict(X_test.values))
    print("Индекс Дэвиса-Болдина:", davies_bouldin)


# оценка количества кластеров
def selection_number_clusters():
    df = data.copy()
    df.drop(['Country', 'location', 'Company Size', 'Job Title', 'Role',
             'skills', 'Company', 'Max Experience', 'Min Salary',
             'Max Salary', 'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day'
             ],
            axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    inertias = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(X_train.values, y_train.values)
        inertias.append(np.sqrt(kmeans.inertia_))
    plt.plot(range(1, 15), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title("Метод локтя")
    plt.savefig('static/charts/ElbowMethod.png')
    plt.close()


# оценка важности параметров
def recursive_feature_elimination():
    df = data.copy()
    df.drop(["Country", "location"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    column_names = ['Qualifications', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
                    'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
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

    estimator = LinearRegression()
    rfe_model = RFE(estimator)
    rfe_model.fit(X_train.values, y_train.values)
    ranks = rank_to_dict_rfe(rfe_model.ranking_, column_names)
    sorted_dict = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))
    print(sorted_dict)


def rank_to_dict_rfe(ranking, names):
    n_ranks = [float(1 / i) for i in ranking]
    n_ranks = map(lambda x: round(x, 2), n_ranks)
    return dict(zip(names, n_ranks))


if __name__ == '__main__':
    # selection_number_clusters()
    # recursive_feature_elimination()
    k_means()
