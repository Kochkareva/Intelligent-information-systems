from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os.path
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


picfld = os.path.join('static', 'charts')

data = pd.read_csv('D:/Интеллектуальные информационные системы/Dataset/updated_job_descriptions.csv')
y = data['Country']


def MLP_classifier_country():
    df = data.copy()
    df.drop(['Country', 'location', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
             'skills', 'Company', 'Min Experience', 'Max Experience', 'Min Salary',
             'Max Salary', 'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day'],
            axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=20)
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    X_train_scaler = scaler.transform(X_train.values)
    X_test_scaler = scaler.transform(X_test.values)
    mlp.fit(X_train_scaler, y_train)
    y_pred = mlp.predict(X_test_scaler)
    precision = precision_score(y_test.values, y_pred, average='weighted')
    recall = recall_score(y_test.values, y_pred, average='weighted')
    print("Precision:", precision)
    print("Recall:", recall)

    # Получаем метки классов
    class_labels = mlp.classes_
    print("Class labels:", class_labels)
    print("Уникальных Country :", data['Country'].nunique())

    # Создаем график
    plt.scatter(X_train['Qualifications'].values, X_train['Work Type'].values, c=y_train.values, cmap='viridis', label='Train Data')
    plt.scatter(X_test['Qualifications'].values, X_test['Work Type'].values, c=y_test.values, cmap='viridis', marker='x', label='Test Data')
    plt.xlabel('Qualifications')
    plt.ylabel('Work Type')
    plt.title('MLPClassifier Visualization')
    plt.savefig('static/charts/MLPClassifier.png')
    plt.close()
    return 0


if __name__ == '__main__':
    MLP_classifier_country()
