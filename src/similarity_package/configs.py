from dataclasses import dataclass
from enum import Enum

import pandas as pd

TARGET_PATIENT_ID = 1
DIAGNOSES_SHEET_NAME = 'פרטים ואבחנות'
LAB_RESULTS_SHEET_NAME = 'בדיקות מעבדה'
DRUGS_SHEET_NAME = 'תרופות'

COMPONENT_STYLE = {
    'box-shadow': '0px 5px 10px 0px rgba(0, 0, 0, 0.5)',
    'margin-top': '10px',
    'margin-right': '10px',
    'margin-left': '10px',
    'align': 'center'}

SHADOW_STYLE = {'box-shadow': '0px 5px 10px 0px rgba(0, 0, 0, 0.5)',
                'margin-top': '10px',
                'margin-right': '10px',
                'margin-left': '10px',
                'align': 'center'
                }

MARGIN_STYLE = {'margin-top': '10px',
                'margin-right': '12px',
                'margin-left': '12px',
                'align': 'center'
                }


@dataclass
class DataView:
    data_source_name: str
    sheet_name: str
    col_name: str


@dataclass
class DataSource:
    name: str
    df: pd.DataFrame
    weight: float

class FeatureType(Enum):
    LST: 1
    CATEGORICAL: 2
    NUMERIC: 3
    TEXT: 4


@dataclass
class Feature:
    feature_name: str
    type: FeatureType
    data_source: DataView


DIAGNOSES = DataView('Diagnosis', DIAGNOSES_SHEET_NAME, 'Diagnosis')
LAB_RESULTS = DataView('Test Results', LAB_RESULTS_SHEET_NAME, 'TestName')
DRUGS = DataView('Drugs', DRUGS_SHEET_NAME, 'DrugName')

DATA_SOURCE_MAPPING = {'Diagnosis': DIAGNOSES, 'Lab Results': LAB_RESULTS,
                       'Drugs': DRUGS}
