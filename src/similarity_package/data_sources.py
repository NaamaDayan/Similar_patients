import math
from abc import ABC
from collections import Counter
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class DataSource(ABC):

    def __init__(self, data: pd.DataFrame, data_source_weight: float):
        self.data_source_weight = data_source_weight
        self.data = data

    def add_features(self, patients_df: pd.DataFrame):
        pass

    def handle_lst_features(self, feature_prefix: str, patients_features: List[pd.Series]):
        transformed_df = pd.DataFrame()
        for patients_feature in patients_features:
            transformed_df[feature_prefix] = patients_feature.apply(
                lambda patient_results: self.calc_similarity_lst_features(patient_results,
                                                                          patients_feature.values[0]))

            transformed_df[f'{feature_prefix}_iou'] = patients_feature.apply(
                lambda patient_results: self.calc_similarity_agg_features(patient_results,
                                                                          patients_feature.values[0]))

        return transformed_df

    def normalize_by_data_source_weight(self, patients_feature_df: pd.DataFrame):
        return patients_feature_df.applymap(lambda val: val * self.data_source_weight)

    def get_column_values_chronologically(self, patient_df: pd.DataFrame, col: str, date_col: str):
        self.sort_df(patient_df, date_col=date_col)
        return list(patient_df[col].values)

    def sort_df(self, patient_df: pd.DataFrame, date_col: str):
        patient_df[date_col] = pd.to_datetime(patient_df[date_col])
        patient_df.sort_values(date_col, inplace=True)

    def get_pair_column_values_chronologically(self, patient_df: pd.DataFrame, col_1: str, col_2: str, date_col: str):
        self.sort_df(patient_df, date_col=date_col)
        return [f'{val_1}_{val_2}' for val_1, val_2 in zip(patient_df[col_1].values, patient_df[col_2].values)]

    def calc_similarity_agg_features(self, lst_1: List[str], lst_2: List[str]) -> float:
        return len(set(lst_1) & set(lst_2)) / len(set(lst_1) | set(lst_2))

    def calc_similarity_lst_features(self, lst_1: List[str], lst_2: List[str]) -> float:
        c1 = Counter(lst_1)
        c2 = Counter(lst_2)
        terms = set(c1).union(c2)
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
        return dotprod / (magA * magB)

    # def get_text_similarity(self) -> List[float]:
    #     """Removes 'stop words' from text and performs TFIDF algorithm.
    #        Returns a list of similarity score based on free text """
    #     vectorizer = TfidfVectorizer(stop_words='english')
    #     texts = self.patients_df.apply(lambda row: " ".join([row[col] for col in self.text_features]), axis=1).values
    #
    #     tfidf_matrix = vectorizer.fit_transform(texts)
    #     return linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()


class Diagnosis(DataSource):

    def add_features(self, patients_df: pd.DataFrame):
        diagnosis_feature = self.data.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'Diagnosis', 'DiagnosisDate'))

        return self.normalize_by_data_source_weight(self.handle_lst_features('diagnosis', [diagnosis_feature]))


class LabResult(DataSource):

    def add_features(self, patients_df: pd.DataFrame):
        lab_test_feature = self.data.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'TestCode', 'ResultDate'))

        lab_test_with_result_feature = self.data.groupby('ID').apply(
            lambda df: self.get_pair_column_values_chronologically(df, 'TestCode', 'ResultType', 'ResultDate'))

        return self.normalize_by_data_source_weight(
            self.handle_lst_features('lab_test_names', [lab_test_feature, lab_test_with_result_feature]))


class Drugs(DataSource):

    def add_features(self, patients_df: pd.DataFrame):
        drugs_feature = self.data.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'DrugName', 'OrderStartDate'))

        drugs_and_status_feature = self.data.groupby('ID').apply(
            lambda df: self.get_pair_column_values_chronologically(df, 'DrugName', 'ExecutionStatus', 'OrderStartDate'))

        return self.normalize_by_data_source_weight(
            self.handle_lst_features('drugs', [drugs_feature, drugs_and_status_feature]))
