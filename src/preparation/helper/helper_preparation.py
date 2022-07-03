import pandas as pd


def eda(path):
    """
    Explorative Data Analysis
    """
    print(path)

    data = pd.read_csv(path)

    print("***** Matrix *****")
    print(pd.crosstab(data['response'], data['treatment'], margins=True))
    print()

    data_treated = data.loc[data.treatment == 1]
    data_control = data.loc[data.treatment == 0]

    print("Number of rows: {} and number of features: {}".format(data.shape[0], data.shape[1]))
    print("Number of treated samples: {}".format(data_treated.shape[0]))
    print("Number of control samples: {}".format(data_control.shape[0]))
    print("Number of treatment responder: {}".format(data_treated.loc[data_treated.response == 1].shape[0]))
    print("Number of treatment non responder: {}".format(data_treated.loc[data_treated.response == 0].shape[0]))
    print("Number of control responder: {}".format(data_control.loc[data_control.response == 1].shape[0]))
    print("Number of control non responder: {}".format(data_control.loc[data_control.response == 0].shape[0]))
    print("Treated response rate: {}".format((data_treated.loc[data_treated.response == 1].shape[0] / data_treated.shape[0])))
    print("Control response rate: {}".format(data_control.loc[data_control.response == 1].shape[0] / data_control.shape[0]))