import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import linear_kernel
from typing import List, Set
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
from collections import Counter


class SimilarityCalculator:

    def __init__(self, patients_df: pd.DataFrame):
        self.patients_df = patients_df
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
        self.lst_features = []

    def get_column_values_chronologically(self, patient_df: pd.DataFrame, col: str, date_col: str):
        self.sort_df(patient_df, date_col=date_col)
        return list(patient_df[col].values)

    def get_pair_column_values_chronologically(self, patient_df: pd.DataFrame, col_1: str, col_2: str, date_col: str):
        self.sort_df(patient_df, date_col=date_col)
        return [f'{val_1}_{val_2}' for val_1, val_2 in zip(patient_df[col_1].values, patient_df[col_2].values)]

    def sort_df(self, patient_lab_tests_df: pd.DataFrame, date_col: str):
        patient_lab_tests_df[date_col] = pd.to_datetime(patient_lab_tests_df[date_col])
        patient_lab_tests_df.sort_values(date_col, inplace=True)

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

    def categorical_distance(self, val_1: str, val_2: str):
        return 1 if val_1 == val_2 else 0

    def get_text_similarity(self) -> List[float]:
        """Removes 'stop words' from text and performs TFIDF algorithm.
           Returns a list of similarity score based on free text """
        vectorizer = TfidfVectorizer(stop_words='english')
        texts = self.patients_df.apply(lambda row: " ".join([row[col] for col in self.text_features]), axis=1).values

        tfidf_matrix = vectorizer.fit_transform(texts)
        return linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()

    def calculate_ranked_similarity(self, feature_scores: pd.DataFrame) -> pd.DataFrame:
        """This function calculates similarity score for every patient in other_patients,
           and returns the original patients_df with a new sorted column of distance.
           Higher similarity means that the patient is closer to the examined patient.
           text_factor is a variable which controlls how much significance we would like to give to the text feature compared to the other features.
           For example, if text_factor = 0.1 (10%), it means that the other features are 10 times more important.
        """
        patient_info = feature_scores.iloc[0].values
        other_patients = feature_scores.iloc[1:]
        distances = distance.cdist([patient_info], other_patients.values, "cosine")[0]
        original_other_patients = self.patients_df.iloc[1:]
        original_other_patients.loc[original_other_patients.index, 'features_similarity'] = 1 / distances
        original_other_patients.loc[
            original_other_patients.index, 'features_similarity'] = MinMaxScaler().fit_transform(
            original_other_patients[['features_similarity']])

        return original_other_patients.sort_values('features_similarity', ascending=False)

    def add_diagnoses_features(self, diagnoses_df: pd.DataFrame):
        self.patients_df['diagnoses'] = diagnoses_df.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'Diagnosis', 'DiagnosisDate'))

    def add_test_results_features(self, lab_tests_df: pd.DataFrame):
        self.patients_df['lab_test_names'] = lab_tests_df.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'TestCode', 'ResultDate'))
        self.patients_df['lab_test_names_and_results'] = lab_tests_df.groupby('ID').apply(
            lambda df: self.get_pair_column_values_chronologically(df, 'TestCode', 'ResultType', 'ResultDate'))

    def add_drugs_features(self, drugs_df: pd.DataFrame):
        self.patients_df['drugs'] = drugs_df.groupby('ID').apply(
            lambda df: self.get_column_values_chronologically(df, 'DrugName', 'OrderStartDate'))
        self.patients_df['drugs_and_status'] = drugs_df.groupby('ID').apply(
            lambda df: self.get_pair_column_values_chronologically(df, 'DrugName', 'ExecutionStatus', 'OrderStartDate'))

    def _split_columns_by_type(self):
        self.numerical_features = self.patients_df.select_dtypes(
            np.number).columns  # TODO:: maybe have default and suggest columns
        self.text_features = list(
            self.patients_df.columns[self.patients_df.applymap(lambda x: isinstance(x, str)).all(0)])
        self.lst_features = list(
            self.patients_df.columns[self.patients_df.applymap(lambda x: isinstance(x, list)).all(0)])
        self.categorical_features = list(
            set(self.patients_df.columns) - set(self.numerical_features) - set(self.text_features) - set(
                self.lst_features))

    def calculate_similarity_per_feature(self) -> pd.DataFrame:
        self._split_columns_by_type()
        scaler = MinMaxScaler()

        transformed_df = self.patients_df[self.numerical_features].copy()
        transformed_df = pd.get_dummies(self.patients_df[self.categorical_features]) if len(
            self.categorical_features) > 0 else transformed_df
        transformed_df[self.numerical_features] = scaler.fit_transform(self.patients_df[self.numerical_features])

        for feature in self.lst_features:
            transformed_df[feature] = self.patients_df[feature].apply(
                lambda patient_results: self.calc_similarity_lst_features(patient_results,
                                                                          self.patients_df[feature].values[0]))

            transformed_df[f'{feature}_iou'] = self.patients_df[feature].apply(
                lambda patient_results: self.calc_similarity_agg_features(patient_results,
                                                                          self.patients_df[feature].values[0]))

        if self.text_features:
            transformed_df['text_similarity'] = self.get_text_similarity()

        return transformed_df


if __name__ == '__main__':
    name = 'קבוצה 1.xlsx'

    diagnoses_df = pd.read_excel(f'/home/naama/Downloads/{name}', sheet_name='קבוצה 1 - פרטים ואבחנות')
    lab_tests_df = pd.read_excel(f'/home/naama/Downloads/{name}', sheet_name='קבוצה 1 - בדיקות מעבדה')
    drugs_df = pd.read_excel(f'/home/naama/Downloads/{name}', sheet_name='קבוצה 1 - תרופות')

    patients_df = diagnoses_df.groupby('ID').mean('birth_year')
    patients_df['text'] = ['The patient is healthy but has penuts alergy',
                           'patient is very healthy and have penuts alergy', 'Unsimilar Text', 'Unsimilar Text alergy',
                           'Unsimilar Text patient alergy', 'Unsimilar Text', 'Unsimilar Text']

    sc = SimilarityCalculator(patients_df)
    sc.add_diagnoses_features(diagnoses_df)
    sc.add_test_results_features(lab_tests_df)
    sc.add_drugs_features(drugs_df)

    feature_scores = sc.calculate_similarity_per_feature()

    sc.calculate_ranked_similarity(feature_scores)

    # from pandas_profiling import ProfileReport
    #
    # prof = ProfileReport(patients_df)
    # prof.to_file(output_file='output.html')
