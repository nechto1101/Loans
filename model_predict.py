import dill
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def predict(datafr):
    with open('models/credit_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    y = model['model'].predict(datafr)

    return {
        'id': datafr['id'][0],
        'pred': y[0],
    }


if __name__ == '__main__':
    for i in range(0, 50):
        df = pd.read_csv(f'test_samples/{i} sample.csv')
        end_fr = predict(df)
        print(end_fr)
