import tensorflow as tf
from kobert.kobert_model import TFBertForSequenceClassification
from kogpt2.kogpt2_model import TFGPT2ForSequenceClassification
import optuna
from packaging import version

if version.parse(tf.__version__) < version.parse("2.0.0"):
    raise RuntimeError("tensorflow>=2.0.0 is required for this example.")

class StratOptimize():
    def __init__(self, config, project_path, data):
        self.config = config
        self.project_path = project_path
        self.data = data
        self.optimize()

    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=2)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.config.lr = trial.params['adam_learning_rate']
        self.config.dropout = trial.params['dropout']

    def objective(self, trial):
        self.model = self.create_model(trial)

        # Training and validating cycle.
        self.learn("train")
        accuracy = self.learn("eval")

        return accuracy

    def create_model(self, trial):
        num_labels = self.data['original']['RCMN_CD1'].nunique()

        dropout = trial.suggest_float("dropout", 0.1, 0.5, log=True)
        learning_rate = trial.suggest_float("adam_learning_rate", 5e-6, 5e-5, log=True)

        if self.config.model == 'kobert':
            if self.config.task == 'binary':
                model = TFBertForSequenceClassification("klue/bert-base", activation='sigmoid', num_labels=1)
                loss = tf.keras.losses.BinaryCrossentropy()
            elif self.config.task == 'multi-class':
                model = TFBertForSequenceClassification("klue/bert-base", activation='softmax',
                                                        num_labels=num_labels)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

        elif self.config.model == 'kogpt2':
            if self.config.task == 'binary':
                model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='sigmoid',
                                                        num_labels=1, dropout=dropout)
                loss = tf.keras.losses.BinaryCrossentropy()
            elif self.config.task == 'multi-class':
                model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='softmax',
                                                        num_labels=num_labels, dropout=dropout)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model

    def learn(self, mode="eval"):
        if mode == "eval":
            accuracy = self.model.evaluate(self.data['test_X'], self.data['test_y'])[1]
            return accuracy
        else:
            history = self.model.fit(self.data['train_X'], self.data['train_y'], epochs=4, batch_size=8,
                                     validation_data=(self.data['valid_X'], self.data['valid_y']))
