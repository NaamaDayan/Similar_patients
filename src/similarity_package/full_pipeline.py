import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.similarity_package.configs import DIAGNOSES_SHEET_NAME, LAB_RESULTS_SHEET_NAME, DRUGS_SHEET_NAME, DataSource
from src.similarity_package.data_sources import Diagnosis, Drugs, LabResult
from src.similarity_package.similarity_calculator import SimilarityCalculator


def get_distance_score_by_group(group_num: int, feature_weight_diagnosis: float, feature_weight_lab_result: float,
                                feature_weight_drugs: float, max_similar_patients: int):

    df_name = f'/home/naama/Downloads/group_{str(group_num)}.xlsx'
    diagnoses_df = pd.read_excel(df_name, sheet_name=DIAGNOSES_SHEET_NAME)
    lab_tests_df = pd.read_excel(df_name, sheet_name=LAB_RESULTS_SHEET_NAME)
    drugs_df = pd.read_excel(df_name, sheet_name=DRUGS_SHEET_NAME)

    patients_df = diagnoses_df.groupby('ID').mean('birth_year')
    # patients_df['text'] = ['The patient is healthy but has penuts alergy',
    #                        'patient is very healthy and have penuts alergy', 'Unsimilar Text', 'Unsimilar Text alergy',
    #                        'Unsimilar Text patient alergy', 'Unsimilar Text', 'Unsimilar Text', 'Unsimilar Text', 'Unsimilar Text'][:len(patients_df)]

    sources = [Diagnosis(diagnoses_df, feature_weight_diagnosis),
               LabResult(lab_tests_df, feature_weight_lab_result),
               Drugs(drugs_df, feature_weight_drugs)]

    transformed_patients_df = pd.concat([data_source.add_features(patients_df) for data_source in sources], axis=1)

    sc = SimilarityCalculator(patients_df)

    ranked_distances = sc.calculate_ranked_distances(transformed_patients_df)
    max_similar_patients = max_similar_patients if isinstance(max_similar_patients, int) else len(ranked_distances)
    print (max_similar_patients)
    print (ranked_distances)
    return ranked_distances.head(max_similar_patients + 1)


def get_data_by_id(group_num: int, patient_id: int, sheet_name):
    df = get_df_by_sheet_name(group_num, sheet_name)
    return df[df['ID'] == int(patient_id)]


def get_df_by_sheet_name(group_num, sheet_name):
    df_name = f'/home/naama/Downloads/group_{str(group_num)}.xlsx'
    return pd.read_excel(df_name, sheet_name=sheet_name)


def get_categories_encodings(label_encoder, df: pd.DataFrame, col_name: str):
    return " ".join(map(str, label_encoder.transform(list(set(df[col_name].values)))))


def get_rare_categories(df: pd.DataFrame, col_name, frequency_factor: float):
    label_encoder = LabelEncoder()
    label_encoder.fit(df[col_name].unique())

    categories_encodings = df.groupby('ID').apply(lambda x: get_categories_encodings(label_encoder, x, col_name)).values
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(categories_encodings)
    terms = vectorizer.get_feature_names()

    data = [(term, X.sum(axis=0)[0, col]) for col, term in enumerate(terms)]
    ranking = pd.DataFrame(data, columns=['term', 'rank'])

    ranking[col_name] = label_encoder.inverse_transform(list(map(int, ranking['term'].values)))
    categories_ranking = ranking.sort_values('rank', ascending=False)
    rank_percentile_threshold = np.percentile(categories_ranking['rank'].values, frequency_factor)
    return categories_ranking[categories_ranking['rank'] <= rank_percentile_threshold]['TestName'].values


def get_shared_categories(df, patient_ids, col_name, frequency_factor=100):
    if frequency_factor < 100:
        rare_categories = get_rare_categories(df, frequency_factor, col_name)
        df = df[df[col_name].isin(rare_categories)]

    shared_categories = []
    patient_1_categories = df[df['ID'] == 1][col_name].value_counts().keys()

    for patient_id in patient_ids:
        patient_categories = df[df['ID'] == patient_id]
        categories = patient_categories[col_name].value_counts().keys()
        shared_categories.append(list(set(categories) & set(patient_1_categories)))

    return pd.DataFrame(shared_categories, index=patient_ids).T


if __name__ == '__main__':
    group_num = 1

    distances = get_distance_score_by_group(1, 0.33, 0.33, 0)['distance']
    print(distances)
