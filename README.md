Использован датасет по прогнозированию количества суицидов
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016

При запуске приложения, по адресу
http://127.0.0.1:5000 
необходимо ввести данные для обработки и предсказания количества суицидов при заданных параметрах

через run_server.py приложение не запускается

if __name__ == '__main__':
    app.run()
сервер запускает, но ссылка выдаёт 404 ошибку, не успел с ней разобраться

### **приложение работает через консльную команду 'flask run' в виртуальном окружении**

_в model / train.py есть баг связанный с импортом класса DataPipeline из файла
model / datapipeline.py. В готовом приложении dill работает только если в train.py прописан импортом по полному пути

from app.model.datapipeline import DataPipeline
Но если в source/dumps файл pipeline.dill отсутсвует, программа запускатеся только с таким импортом  

from datapipeline import DataPipeline_

### **Содержимое проекта:**

app/model/datapipeline.py - пайплайн
    Использован классификатор RandomForestRegressor с значениями по умолчанию
    RandomForestRegressor
    (OHEEncoderBin, используемый на уроке на работает для одного объекта, заставить зараотать не получилось)

app/model/predict.py - тестовое предсказание и расчет метрик
    предсказания записаны в файл source/prediction/test_predictions.csv
    полученные метрики записаны в файл source/prediction/metrics.txt

app/model/train.py - обучение модели и запись пайплайна в файл source/dumps/pipeline.dill

app/templates - каталог с шаблонами html-страниц

app/__init__.py - инициализация приложения

app/errors.py - обработка ошибок (404)

app/routes.py - файл с функциями для адресных путей 

source/train_test - каталог, содержащий train/test выборки

source/master.csv - датасет

source/master_info.txt - датасет инфо

config.py - файл конфигурации, содержащий пути к ресурсам

run_server.py - запуск приложения (выдаёт ошибку 404)
