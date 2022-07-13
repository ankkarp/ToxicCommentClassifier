# ToxicCommentClassifier
 
## Запуск:

### Инференс

from ToxicCommentClassifier.inference import Tester


tester = Tester()

tester.load_model('./e2_loss0.1510_f1_score_0_91563.h5')

tester.test(test_csv='data_test_public[494].csv', export_file='res.csv')  # тестирование модели

tester.get_mistakes(print_accs=True)

### Обучение (и инференс)

from ToxicCommentClassifier.train import Trainer

trainer = Trainer()

model = trainer.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv')

Также объект тренировщика наследует все методы инференса, то есть можно сразу вызывать эти методы для инференса только что обученной модели.

trainer.test(test_csv='data_test_public[494].csv', export_file='res.csv')  # тестирование модели

trainer.get_mistakes(print_accs=True)

Для обучения новой нет необходимости создавать новый объект класса Trainer, достаточно просто вызвать метод train, так как

## Обученная модель

Ссылка на модель: https://drive.google.com/file/d/1xU9p1TTLU3_sFSF73NXMvk9t7N4HJEOJ/view?usp=sharing

f1 score: 0.91563
CrossEntropyLoss = 0.151

## Функционал

### Класс инференса Tester

#### Конструктор 

Tester(model=None, vocab='DeepPavlov/rubert-base-cased-conversational', token_len=128)

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| model | модель, которую нужно протестировать | Object | None
| vocab | название модели/словаря bert | str | "DeepPavlov/rubert-base-cased-conversational"
| token_len | длина последовательностей токенов | int | 128

#### Метод инфернеса

test(test_csv: str, batch_sz=16, export_file=None) -> pd.DataFrame

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| test_csv | путь к csv-файлу с данными на инференс | str |
| batch_sz | размер батча | int | 16
| export_file | путь к csv файлу, в который нужно сохранить результат | str | None

Возвращает датафрейм с данными инференса

#### Метод для получения датафрейма данных, которые были классифицированы неправильно

get_mistakes(self, export_file=None, print_accs=False) -> pd.DataFrame

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| export_file | путь к csv файлу, в который нужно сохранить результат | str | None
| print_accs | вывести в консоль точность модели по каждому классу | bool | False

Возвращает таблицу данных, на которых модель ошиблась

#### Метод загрузки модели в тестер

load_model(self, path: str) -> None

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| path | путь к файлу модели | str | |

### Класс обучения (и инференса Trainer)

Trainer - потомок класса Tester и наследует все его методы и поля

#### Конструктор 

Tester(model_class=BertForSequenceClassification, vocab='DeepPavlov/rubert-base-cased-conversational', token_len=128)

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| model_class | класс модели (BertForSequenceClassification или BertClassifier) | class | BertForSequenceClassification
| vocab | название модели/словаря bert | str | "DeepPavlov/rubert-base-cased-conversational"
| token_len | длина последовательностей токенов | int | 128

#### Метод обучения

train(train_csv: str, val_csv: str, lr=5e-6, epochs=4, batch_sz=16, name='ToxicCommentClassifier', balance=False, balance_sz=None, get_best=True, wandb_logging=False)

| Параметр | Описание  | Тип данных | Значение по умолчанию
| ------------- |:-------------:|:-------------:|:-------------:| 
| test_csv | путь к csv-файлу с данными для обучения | str |
| test_csv | путь к csv-файлу с данными для валидации | str |
| lr | learning rate для оптимизатора Adam | float | 5e-6
| batch_sz | размер батча | int | 16
| name | имя проекта (для wandb и логирования промежуточных моделей) | str | False
| balance | вернуть лучшую модель | bool | True
| balance_sz | азмер, до которого нужно уменьшить датасеты (при balance=True) | int | None
| wandb_logging | логировать в wandb | bool | True

Возвращает таблицу результата инференса
