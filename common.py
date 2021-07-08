from sklearn.metrics import f1_score, classification_report, accuracy_score


class Common:
    @staticmethod
    def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
        """
        Evaluates a model's performance on predicting x_test, where y_test is truth, after training with
        x_train & y_train; print out results
        :param model: can be anything e.g. LogisticRegression(), DecisionTreeClassifier()
        :param x_train: train feature space
        :param y_train: ground truth of train set
        :param x_test: test feature space
        :param y_test: ground truth of test set
        :param model_name: printed with f1 score and accuracy, for readability
        """
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("f1 target", model_name, f1)
        print("acc target", model_name, acc)
        print(report)

        return f1
