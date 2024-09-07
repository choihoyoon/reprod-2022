from utils import create_directory
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import json
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class StratTrain():
    def __init__(self, config, for_card, project_path, data, model):
        self.config = config
        self.for_card = for_card
        self.project_path = project_path
        self.data = data
        self.model = model
        self.run_experiment()

    class create_learning_curve(Callback):
        def __init__(self, project_path):
            self.project_path = project_path
            self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        def on_epoch_end(self, epoch, logs=None):
            self.history['loss'].append(logs.get('loss'))
            self.history['accuracy'].append(logs.get('accuracy'))
            self.history['val_loss'].append(logs.get('val_loss'))
            self.history['val_accuracy'].append(logs.get('val_accuracy'))
            StratTrain.learning_curve(self, self.history)

    def run_experiment(self):
        mc_path = self.project_path + '/check_point'
        create_directory(mc_path)

        mc_path = mc_path + '/epoch_{epoch:02d}'
        mc = ModelCheckpoint(mc_path, verbose=1, save_weights_only=True)

        history = self.model.fit(self.data['train_X'], self.data['train_y'], epochs=self.config.n_epochs,
                                 batch_size=self.config.batch_size,
                                 validation_data=(self.data['valid_X'], self.data['valid_y']),
                                 callbacks=[mc, self.create_learning_curve(self.project_path)])
        print(self.model.summary())

        history_json = json.dumps(history.history, ensure_ascii=False, indent=4)
        history_file = open(self.project_path + '/history.json', 'w')
        print(history_json, file=history_file)
        history_file.close()

        results = self.model.evaluate(self.data['test_X'], self.data['test_y'])
        print("test loss, test acc: ", results)

        result = history.history
        metric = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        for i in metric:
            result[i] = list(result[i])[-1]
        result['test_loss'] = results[0]
        result['test_accuracy'] = results[1]
        result_json = json.dumps(result, ensure_ascii=False, indent=4)
        result_file = open(self.project_path + '/result.json', 'w')
        print(result_json, file=result_file)
        result_file.close()

        label = np.unique(self.data['test_y']).tolist()
        y_pred = self.model.predict(self.data['test_X'])
        y_pred = np.argmax(y_pred, axis=1)
        conf_mat = confusion_matrix(self.data['test_y'], y_pred, normalize='true')
        conf_plot = px.imshow(conf_mat, text_auto=True, labels=dict(x="Predicted label", y="True label"),
                              x=label, y=label)
        conf_file = open(self.project_path + '/plot/conf_mat.json', 'w')
        print(conf_plot.to_json(), file=conf_file)
        conf_file.close()

        report_file = open(self.project_path + '/report.log', 'w')
        print(classification_report(self.data['test_y'], y_pred), file=report_file)
        report_file.close()

        self.model.save_weights(self.project_path + '/final_model')
        self.for_card['result'] = result

    def learning_curve(self, history):
        plot_path = self.project_path + '/plot'
        create_directory(plot_path)

        acc_file = open(plot_path + '/accuracy.json', 'w')
        accuracy = px.line(history['accuracy'], title="accuracy")
        print(accuracy.to_json(), file=acc_file)
        acc_file.close()

        val_acc_file = open(plot_path + '/val_accuracy.json', 'w')
        val_accuracy = px.line(history['val_accuracy'], title="val_accuracy")
        print(val_accuracy.to_json(), file=val_acc_file)
        val_acc_file.close()

        loss_file = open(plot_path + '/loss.json', 'w')
        loss = px.line(history['loss'], title="loss")
        print(loss.to_json(), file=loss_file)
        loss_file.close()

        val_loss_file = open(plot_path + '/val_loss.json', 'w')
        val_loss = px.line(history['val_loss'], title="val_loss")
        print(val_loss.to_json(), file=val_loss_file)
        val_loss_file.close()
