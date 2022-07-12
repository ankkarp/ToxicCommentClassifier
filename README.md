# ToxicCommentClassifier
 
## Запуск инференса:

from ToxicCommentClassifier.inference import Tester


tester = Tester()

tester.load_model('./e2_loss0.1510_f1_score_0_91563.h5')

tester.test(test_csv='data_test_public[494].csv', export_file='res.csv')  # тестирование модели

tester.get_mistakes(print_accs=True)

## Запуск обучения

from ToxicCommentClassifier.train import Trainer

trainer = Trainer()

model = trainer.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv')

Также объект тренировщика наследует все методы инференса

Ссылка на модель: https://drive.google.com/file/d/1xU9p1TTLU3_sFSF73NXMvk9t7N4HJEOJ/view?usp=sharing
