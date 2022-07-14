import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler


class SimilarityCalculator:

    def __init__(self, patients_df: pd.DataFrame):
        self.patients_df = patients_df
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
        self.lst_features = []

    def categorical_distance(self, val_1: str, val_2: str):
        return 1 if val_1 == val_2 else 0

    def calculate_ranked_distances(self, feature_scores: pd.DataFrame) -> pd.DataFrame:
        """This function calculates similarity score for every patient in other_patients,
           and returns the original patients_df with a new sorted column of distance.
           Higher similarity means that the patient is closer to the examined patient.
           text_factor is a variable which controlls how much significance we would like to give to the text feature compared to the other features.
           For example, if text_factor = 0.1 (10%), it means that the other features are 10 times more important.
        """
        patient_info = feature_scores.iloc[0].values
        other_patients = feature_scores
        original_other_patients = self.patients_df
        original_other_patients['distance'] = distance.cdist([patient_info], other_patients.values, "cosine")[0]

        original_other_patients['distance'] = MinMaxScaler().fit_transform(
            original_other_patients[['distance']])

        return original_other_patients.sort_values('distance', ascending=True)

    # def _split_columns_by_type(self):
    #     self.numerical_features = self.patients_df.select_dtypes(np.number).columns
    #     self.text_features = list(
    #         self.patients_df.columns[self.patients_df.applymap(lambda x: isinstance(x, str)).all(0)])
    #     self.lst_features = list(
    #         self.patients_df.columns[self.patients_df.applymap(lambda x: isinstance(x, list)).all(0)])
    #     self.categorical_features = list(
    #         set(self.patients_df.columns) - set(self.numerical_features) - set(self.text_features) - set(
    #             self.lst_features))

    # def calculate_similarity_per_feature(self) -> pd.DataFrame:
    #     self._split_columns_by_type()
    #     scaler = MinMaxScaler()
    #
    #     transformed_df = self.patients_df[self.numerical_features].copy()
    #     transformed_df = pd.get_dummies(self.patients_df[self.categorical_features]) if len(
    #         self.categorical_features) > 0 else transformed_df
    #     # transformed_df[self.numerical_features] = scaler.fit_transform(self.patients_df[self.numerical_features])
    #     # {data_source: (feature, type)}
    #     for feature in self.lst_features:
    #         transformed_df[feature] = self.patients_df[feature].apply(
    #             lambda patient_results: self.calc_similarity_lst_features(patient_results,
    #                                                                       self.patients_df[feature].values[0]))
    #
    #         transformed_df[f'{feature}_iou'] = self.patients_df[feature].apply(
    #             lambda patient_results: self.calc_similarity_agg_features(patient_results,
    #                                                                       self.patients_df[feature].values[0]))
    #
    #     if self.text_features:
    #         transformed_df['text_similarity'] = self.get_text_similarity()
    #
    #     return transformed_df
