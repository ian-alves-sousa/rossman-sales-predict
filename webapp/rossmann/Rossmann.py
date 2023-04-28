import pickle
import inflection
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime, timedelta


class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        with open(self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb') as RS:
            self.competition_distance_scaler = pickle.load(RS)
        with open(self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb') as RS1:
            self.competition_time_month_scaler = pickle.load(RS1)
        with open(self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb') as MMS1:
            self.promo_time_week_scaler = pickle.load(MMS1)
        with open(self.home_path + 'parameters/year_scaler.pkl', 'rb') as MMS2:
            self.year_scaler = pickle.load(MMS2)
        with open(self.home_path + 'parameters/store_type_scaler.pkl', 'rb') as LE:
            self.store_type_scaler = pickle.load(LE)

    def data_cleaning(self, df1):
        """Recebe um df1 e limpa ele"""
        # 1.1 Rename Columns

        # Colocando como snake case
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        def snake_case(x): return inflection.underscore(x)
        cols_new = list(map(snake_case, cols_old))

        # Rename
        df1.columns = cols_new
        df1.columns

        # 1.3 Data Types

        # Muda o tipo do date de object para data
        df1['date'] = pd.to_datetime(df1['date'])

        # 1.5 Fillout NA

        # competition_distance              2642
        # Vamos completar essa coluna com um valor muito mais alto que o maior valor da caluna, pois para ela ser NA
        # Essa loja não tem um competidor próximo, então colocamos um alto valor para representar isso
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x: 200000 if math.isnan(x) else x)

        # competition_open_since_month    323348
        # Nesse caso, vamos utilizar a data que temos como parâmetro para essa data
        # Derivamos o mes de date para o início da data de competição
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(
            x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year     323348
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(
            x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # promo2_since_week               508031
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(
            x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year               508031
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(
            x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # promo_interval                  508031
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['promo_interval'].fillna(0, inplace=True)

        df1['month_map'] = df1['date'].dt.month.map(month_map)

        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(
            lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # 1.6 Change Types

        # Depois de alterar os dados das colunas é bom conferir os tipos de ajustá-los
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(
            'int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(
            'int64')

        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')

        return df1

    def feature_engeneering(self, df2):

        # Variáveis para devivar de date
        # year, month, day, week of year, year week
        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.strftime("%U")
        df2['year-week'] = df2['date'].dt.strftime('%Y-%W')
        df2['year'] = df2['year'].astype('int64')
        df2['month'] = df2['month'].astype('int64')
        df2['day'] = df2['day'].astype('int64')
        df2['week_of_year'] = df2['week_of_year'].astype('int64')

        # Variáveis de competição
        # competition since, promo since
        # Juntar o ano da competição com o mês para mostrar o tempo da competição ativa por mês
        df2['competition_since'] = df2.apply(lambda x: datetime(
            year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df2['competition_time_month'] = (
            (df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype('int64')

        # Tempo que minha competição está ativa em semanas
        df2['promo_since'] = df2['promo2_since_year'].astype(
            str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.strptime(x + '-1', '%Y-%W-%w') - timedelta(days=7))
        df2['promo_time_week'] = (
            (df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype('int64')

        # Variáveis que estão com letras e quero trocar por nomes
        # assostment,state holiday
        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public_holiday' if x == 'a' else 'easter_holday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # 3.0 Filtragem de variáveis

        # 3.1 Filtragem das linhas

        # open não tem aprendizado, pois quando está fechado não influencia na quantidade de vendas, precisa ser diferente de 0
        # sales precisa ser maior que zero, para não considerar os dias fechados

        df2 = df2[(df2['open'] != 0)]

        # 3.2 Seleção das Colunas

        # custumers não tem como usar no momento da predição, pois não tem como saber quantos clientes vão ter na loja para influenciar na venda

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    def data_preparation(self, df5):

        # 5.2 Rescaling
        # Como já temos o modelo pronto e carregado, utilizamos o transform ao invés do fit_transform
        # 'competition_distance' tem outliers, vamos usar o robust scale
        df5['competition_distance'] = self.competition_distance_scaler.transform(
            df5[['competition_distance']].values)

        # 'competition_time_month' tem outliers, vamos usar o robust scale
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(
            df5[['competition_time_month']].values)

        # promo_time_week não tem muitos outliers
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(
            df5[['promo_time_week']].values)

        # year não é cíclico e não tem muitos outliers
        df5['year'] = self.year_scaler.transform(df5[['year']].values)

        # 5.3.1 Encoding

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(
            df5, prefix=['state_holiday'], dtype='int64', columns=['state_holiday'])

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # 5.3.3 Nature Transformation

        # month - Ciclo de 12 meses
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

        # day - Ciclo de 31
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

        # week_of_year - Ciclo de 53
        df5['week_of_year_sin'] = df5['week_of_year'].apply(
            lambda x: np.sin(x*(2*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(
            lambda x: np.cos(x*(2*np.pi/52)))

        # day_of_week - Ciclo de 7
        df5['day_of_week_sin'] = df5['day_of_week'].apply(
            lambda x: np.sin(x*(2*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(
            lambda x: np.cos(x*(2*np.pi/7)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                         'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week',
                         'month_cos', 'month_sin', 'day_sin', 'day_cos', 'day_of_week_sin', 'day_of_week_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')
