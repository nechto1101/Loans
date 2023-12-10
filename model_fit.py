import os
import tqdm
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

import dill as dill

import warnings
warnings.filterwarnings("ignore")


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0, num_parts_to_read: int = 2,
                                    columns=None, verbose=False) -> pd.DataFrame:
    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('train')])

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        print('chunk_path', chunk_path)
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)


def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                 num_parts_total: int = 50,
                                 save_to_path=None, verbose: bool = False):
    preprocessed_frames = []

    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                             verbose=verbose)
        #  Понижение разрядности для меньшего веса
        for col in transactions_frame.drop('id', axis=1).columns:
            transactions_frame[col] = transactions_frame[col].astype('int8')
        transactions_frame['id'] = transactions_frame['id'].astype('int32')

        preprocessed_frames.append(transactions_frame)
    return pd.concat(preprocessed_frames)


def filter_data(df) -> pd.DataFrame:  # Удаление ненужного столбца
    import pandas as pd
    df = df
    columns_to_drop = ['pre_loans_total_overdue']
    return df.drop(columns_to_drop, axis=1)


def npl_changing(datafr) -> pd.DataFrame:
    import pandas as pd
    datafr = datafr

    # Вычисление вероятности просрочки в каждой группе по флагам просрочки
    def npl_all_buckets(datafr, c_npl, m_npl) -> pd.DataFrame:
        import pandas as pd
        import numpy as np
        datafr = datafr

        # Группы с нулевой просрочкой
        buckets = {'pre_loans5': 6,
                   'pre_loans530': 16,
                   'pre_loans3060': 5,
                   'pre_loans6090': 4,
                   'pre_loans90': 8}

        all_id = datafr.groupby('id')['id'].count().reset_index(name='all_count')

        # Вычисление процента по каждой группе
        for i in range(len(c_npl)):
            zero = (datafr[(datafr[c_npl[i]] == buckets[c_npl[i]]) & (datafr[m_npl[i]] == 1)].groupby('id')[c_npl[i]].
                    count().reset_index(name='count_0'))
            all_id = all_id.merge(zero[['id', 'count_0']], on='id', how="outer")

            name_out = 'perc_' + c_npl[i]
            all_id[name_out] = round(1 - all_id['count_0'] / all_id['all_count'], 2).astype('float32')
            all_id.drop('count_0', axis=1, inplace=True)

        all_id.fillna(np.float32(1), inplace=True)
        end_frame = all_id.drop(['all_count'], axis=1)
        return end_frame

    counts_npl = ['pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90']
    mark_npl = ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']
    col_for_npl = counts_npl + mark_npl + ['id']

    npl_frame = npl_all_buckets(datafr[col_for_npl], counts_npl, mark_npl)

    out_frame = datafr.merge(npl_frame, on='id')
    out_frame = out_frame.drop(mark_npl, axis=1)

    return out_frame


def ending_preobr(datafr) -> pd.DataFrame:  # Для суммирования по категориальным признакам по клиенту
    import pandas as pd
    datafr = datafr
    # Создание списка с процентами перехода для добавления после аггрерирования категориальных признаков
    col_for_adding = ['id', 'perc_pre_loans5', 'perc_pre_loans530', 'perc_pre_loans3060', 'perc_pre_loans6090',
                      'perc_pre_loans90']
    npl_for_adding = datafr[col_for_adding].drop_duplicates()

    datafr.drop('rn', axis=1)
    rez_fr = datafr.groupby("id")[datafr.drop(npl_for_adding, axis=1).columns].sum().reset_index(drop=False)

    out_frame = rez_fr.merge(npl_for_adding, on='id')
    return out_frame.drop('id', axis=1)


def fit() -> None:
    path = 'train_data/'

    x = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=3, num_parts_total=3, save_to_path=path)
    y = pd.read_csv('train_target.csv', usecols=['flag'], nrows=750000)

    # Столбцы для OHE-преобразования
    categorical_features = x.drop(['id', 'rn', 'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060',
                                   'is_zero_loans6090', 'is_zero_loans90', 'pre_loans_total_overdue'], axis=1).columns
    # Столбцы для исключения столбцов с произведённой просрочкой (не требуемой OHE-преобразования)
    remaining_features = ['id', 'rn', 'perc_pre_loans5', 'perc_pre_loans530', 'perc_pre_loans3060',
                          'perc_pre_loans6090', 'perc_pre_loans90']

    changes_before_ohe = Pipeline(steps=[
        ('deletion', FunctionTransformer(filter_data)),  # Удаляет ненужные столбцы
        ('npl_trans', FunctionTransformer(npl_changing))  # Считает процент просрочки договоров в каждой группе
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse=False, dtype=np.int8, handle_unknown='ignore'), categorical_features)],
        remainder='passthrough', verbose_feature_names_out=True
    )  # OHE-кодировка

    changes_after_ohe = Pipeline(steps=[
        ('convert_to_df', FunctionTransformer(lambda X: pd.DataFrame(X, columns=(list(preprocessor.named_transformers_[
             'cat'].get_feature_names_out(categorical_features)) + list(remaining_features))), validate=False)),
        ('summary', FunctionTransformer(ending_preobr))  # Схлопываение по id клиента
    ])

    model = GradientBoostingClassifier(min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt',
                                       learning_rate=0.005, max_depth=7, n_estimators=1750)

    pipe = Pipeline(steps=[
        ('first_changes', changes_before_ohe),
        ('ohe', preprocessor),
        ('second_changes', changes_after_ohe),
        ('classifier', model)
    ])

    pipe.fit(x, y)

    with open('models/credit_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Defolt_Loans',
                'author': 'Valentin',
                'version': 1,
                'type': type(pipe.named_steps["classifier"]).__name__,
            }
        }, file)


if __name__ == '__main__':
    fit()
