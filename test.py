from BertClassifier import BertClassifier
from inference import Tester
from train import Trainer

if __name__ == "__main__":
    model = Trainer(model_class=BertClassifier, token_len=32, batch_sz=2)
    # model.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv', balance=True, balance_sz=2)
    model.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv', balance=True)
    # print(model.test(test_csv='data_test_public[494].csv', export_file='res.csv'))
    model.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv')
    # tester = Tester(batch_sz=32, token_len=2)
    # tester.load_model('ToxicCommentClassifier/e4_loss0.5716.h5')
    # print(tester.test(test_csv='data_test_public[494].csv', export_file='res.csv', analyse=True))