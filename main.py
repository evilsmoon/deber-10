from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)
# settings
app.secret_key = "mysecretkey"


def cargariris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return X_train, X_test, y_train, y_test, classes


dataset = pd.read_csv("dataset.csv", encoding="utf8", delimiter=",")


def cargaCelular():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    classes = {0: 'sobrevive', 1: 'sobrevive', 2: 'No sobrebive'}
    return X_train, X_test, y_train, y_test, classes


dataset = pd.read_csv("dataset.csv", encoding="utf8", delimiter=",")


# def cargaCelular():
#    y=dataset.iloc[:,-1]
#   X=dataset
#  X=X.drop(['price_range'], axis=1)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)
# classes = {0:'80-200 $',1:'201-500 $',2:'501-800 $',3:'801-1100 $'}
# return X_train,X_test,y_train,y_test,classes

def Knn(X_train, X_test, y_train, y_test, classes, x_new):
    k_range = range(1, 26)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    n_neighbors = scores_list.index(max(scores_list))
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accuracyKNN = metrics.accuracy_score(y_test, y_predict)
    print('------  KNN  -------\n')
    print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))
    print('Accuracy: ', accuracyKNN)
    y_predict = knn.predict(x_new)
    resultado = []
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyKNN, resultado


def randRandom_Forest(X_train, X_test, y_train, y_test, classes, x_new):
    # random_state=42
    # X_train,X_test,y_train,y_test,classes=cargariris(random_state)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('------  randRandom_Forest  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyRF = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracyRF)
    y_predict = clf.predict(x_new)
    resultado = []
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyRF, resultado


def Naive_Bayes(X_train, X_test, y_train, y_test, classes, x_new):
    # random_state=42
    # X_train,X_test,y_train,y_test,classes=cargariris(random_state)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print('------  Naive_Bayes  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyNB = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracyNB)
    y_predict = gnb.predict(x_new)
    resultado = []
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyNB, resultado


def Regresion_logistica(X_train, X_test, y_train, y_test, classes, x_new):
    # random_state=None
    # X_train,X_test,y_train,y_test,classes=cargariris(random_state)
    algoritmo = LogisticRegression()
    algoritmo.fit(X_train, y_train)
    y_pred = algoritmo.predict(X_test)
    print('------  Regresion_logistica  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracyRL = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracyRL)
    y_predict = algoritmo.predict(x_new)
    resultado = []
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracyRL, resultado


def SVM(X_train, X_test, y_train, y_test, classes, x_new):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print('------  SVM  -------\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracySVM = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracySVM)
    y_predict = svclassifier.predict(x_new)
    resultado = []
    for i in range(len(y_predict)):
        resultado.append(classes[y_predict[i]])
    return accuracySVM, resultado


@app.route('/')
def Index():
    return render_template('index.html')


@app.route('/Iris', methods=['POST'])
def add_queryquery():
    if request.method == 'POST':
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
        x_new_iris = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]
        X_train, X_test, y_train, y_test, classes = cargariris()

        accuracyKNN, resultadoknn = Knn(X_train, X_test, y_train, y_test, classes, x_new_iris)
        accuracyRF, resultadoRF = randRandom_Forest(X_train, X_test, y_train, y_test, classes, x_new_iris)
        accuracyNB, resultadoNB = Naive_Bayes(X_train, X_test, y_train, y_test, classes, x_new_iris)
        accuracyRL, resultadoRL = Regresion_logistica(X_train, X_test, y_train, y_test, classes, x_new_iris)
        accuracySVM, resultadoSVM = SVM(X_train, X_test, y_train, y_test, classes, x_new_iris)

        # print(accuracyNB)
        # print(resultadoNB)
        return render_template('consulta_Iris.html', sepal_length=sepal_length,
                               sepal_width=sepal_width,
                               petal_length=petal_length,
                               petal_width=petal_width,
                               accuracyKNN=accuracyKNN,
                               resultadoknn=resultadoknn,
                               accuracyRF=accuracyRF,
                               resultadoRF=resultadoRF,
                               accuracyNB=accuracyNB,
                               resultadoNB=resultadoNB,
                               accuracyRL=accuracyRL,
                               resultadoRL=resultadoRL,
                               accuracySVM=accuracySVM,
                               resultadoSVM=resultadoSVM)


@app.route('/haberman', methods=['POST'])
def ml_celular():
    if request.method == 'POST':
        nucleos = request.form['nucleos']
        reloj = request.form['reloj']
        ram = request.form['ram']
        rom = request.form['rom']
        x_new_celular = [[float(nucleos), float(reloj), float(ram), float(rom)]]
        X_train, X_test, y_train, y_test, classes = cargaCelular()

        accuracyKNN, resultadoknn = Knn(X_train, X_test, y_train, y_test, classes, x_new_celular)
        accuracyRF, resultadoRF = randRandom_Forest(X_train, X_test, y_train, y_test, classes, x_new_celular)
        accuracyNB, resultadoNB = Naive_Bayes(X_train, X_test, y_train, y_test, classes, x_new_celular)
        accuracyRL, resultadoRL = Regresion_logistica(X_train, X_test, y_train, y_test, classes, x_new_celular)
        accuracySVM, resultadoSVM = SVM(X_train, X_test, y_train, y_test, classes, x_new_celular)

        # print(accuracyNB)
        # print(resultadoNB)
        return render_template('consulta_Celular.html', nucleos=nucleos,
                               accuracyKNN=accuracyKNN,
                               resultadoknn=resultadoknn,
                               accuracyRF=accuracyRF,
                               resultadoRF=resultadoRF,
                               accuracyNB=accuracyNB,
                               resultadoNB=resultadoNB,
                               accuracyRL=accuracyRL,
                               resultadoRL=resultadoRL,
                               accuracySVM=accuracySVM,
                               resultadoSVM=resultadoSVM)


if __name__ == '__main__':
    app.run(debug=True)
