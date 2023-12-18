import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main():
    pd.options.display.max_columns = 50
    df = pd.read_csv("cs-training.csv")
    test_df = pd.read_csv("cs-test.csv")

    print(df.SeriousDlqin2yrs.mean())
    print(df.DebtRatio.describe())
    print(df.DebtRatio.quantile([.975]))
    drop_debt = df[df['DebtRatio'] > 3489.025].index
    df.drop(drop_debt, inplace=True)

    drop_lines = df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index
    df.drop(drop_lines, inplace=True)

    df.drop(columns=df.columns[0], axis=1,  inplace=True)
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)
    # print(train_df.isnull().any())
    df.fillna({"MonthlyIncome":df["MonthlyIncome"].median(),
                     "NumberOfDependents": 0}, inplace=True)

    print(df.isnull().any())


    # dim = df.shape[1]
    # fig, ax = plt.subplots(dim, dim, figsize=(16, 9))
    # for i in range(dim):
    #     for j in range(dim):
    #         if i != j:
    #             ax[i, j].scatter(df.iloc[:, i], df.iloc[:, j], s=5, edgecolors='#000099')
    #         else:
    #             ax[i, j].hist(df.iloc[:, i], bins=8, color='orange')
    #         if j == 0:
    #             ax[i, j].set_ylabel(df.columns[i], rotation=0, horizontalalignment='right')
    #         if i == dim - 1:
    #             ax[i, j].set_xlabel(df.columns[j])
    #         ax[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # fig.suptitle('Scatterplot matrix')
    # plt.tight_layout()
    # plt.savefig("Scatterplot")

    cm = df.corr()
    print(cm)

    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns = "SeriousDlqin2yrs")
    standard = StandardScaler()  # scaling the data using standard scaler
    X_t = standard.fit_transform(X)  # transforming here
    X = pd.DataFrame(X_t, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

    model = MLPClassifier(hidden_layer_sizes=(20,20,20), activation="relu", max_iter=50, alpha=1e-3,
                          solver="adam", learning_rate_init=.01, verbose=True, random_state=2023)

    model.fit(X_train, y_train)

    # Plot the loss curve.
    plt.plot(model.loss_curve_)
    plt.show()

    # Display the accuracy of your model.
    print("Accuracy is: ", model.score(X_test, y_test))

    # Display the confusion matrix
    y_pred = model.predict(X_test)
    cf = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cf).plot()
    plt.show()

    test_df.fillna({"MonthlyIncome":test_df["MonthlyIncome"].median(),
                     "NumberOfDependents": 0}, inplace=True)
    print(test_df.isnull().any())
    test = test_df.drop(columns = "SeriousDlqin2yrs")
    standard.fit(test)  # transforming here
    test = pd.DataFrame(standard.transform(test), columns=test.columns)
    pred = model.predict(test)  # running the prediction
    print("Outcome Prediction:", pred)  # printing

if __name__ == '__main__':
    main()