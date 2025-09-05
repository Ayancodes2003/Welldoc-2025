"""
Dynamic Data Loader for Risk Prediction Engine

Automatically detects and adapts to different dataset structures.
Supports MIMIC-IV, eICU, synthetic data, or custom formats.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters"""
    
    @abstractmethod
    def detect_format(self, data_path: str) -> bool:
        """Detect if this adapter can handle the given dataset"""
        pass
    
    @abstractmethod
    def load_and_standardize(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load data and return in standardized format"""
        pass
    
    @abstractmethod
    def get_feature_config(self) -> Dict[str, Any]:
        """Return feature configuration for this dataset type"""
        pass

class MIMICAdapter(DatasetAdapter):
    """Adapter for MIMIC-IV dataset"""
    
    def detect_format(self, data_path: str) -> bool:
        """Check if this looks like MIMIC-IV data"""
        path = Path(data_path)
        
        # Look for typical MIMIC-IV files
        mimic_files = ['patients.csv', 'admissions.csv', 'chartevents.csv', 'labevents.csv']
        found_files = [f for f in mimic_files if (path / f).exists()]
        
        return len(found_files) >= 2  # At least 2 MIMIC files present
    
    def load_and_standardize(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load MIMIC-IV data and standardize column names"""
        path = Path(data_path)
        
        datasets = {}
        
        # Load patients
        if (path / 'patients.csv').exists():
            patients = pd.read_csv(path / 'patients.csv')
            datasets['patients'] = self._standardize_patients(patients)
        
        # Load admissions
        if (path / 'admissions.csv').exists():
            admissions = pd.read_csv(path / 'admissions.csv')
            datasets['admissions'] = self._standardize_admissions(admissions)
        
        # Load vitals from chartevents
        if (path / 'chartevents.csv').exists():
            chartevents = pd.read_csv(path / 'chartevents.csv')
            datasets['vitals'] = self._extract_vitals(chartevents)
        
        # Load labs from labevents
        if (path / 'labevents.csv').exists():
            labevents = pd.read_csv(path / 'labevents.csv')
            datasets['labs'] = self._extract_labs(labevents)
        
        logger.info(f"Loaded MIMIC-IV dataset with {len(datasets)} tables")
        return datasets
    
    def _standardize_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize patient demographics"""
        column_mapping = {
            'subject_id': 'patient_id',
            'gender': 'gender',
            'anchor_age': 'age',
            'dod': 'death_date'
        }
        
        standardized = df.rename(columns=column_mapping)
        
        # Ensure patient_id is string
        if 'patient_id' in standardized.columns:
            standardized['patient_id'] = standardized['patient_id'].astype(str)
        
        return standardized
    
    def _standardize_admissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize admissions data"""
        column_mapping = {
            'subject_id': 'patient_id',
            'hadm_id': 'admission_id',
            'admittime': 'admission_date',
            'dischtime': 'discharge_date',
            'los': 'length_of_stay'
        }
        
        standardized = df.rename(columns=column_mapping)
        
        # Convert dates
        date_columns = ['admission_date', 'discharge_date']
        for col in date_columns:
            if col in standardized.columns:
                standardized[col] = pd.to_datetime(standardized[col])
        
        return standardized
    
    def _extract_vitals(self, chartevents: pd.DataFrame) -> pd.DataFrame:
        """Extract vital signs from chartevents"""
        # MIMIC-IV vital sign item IDs (common ones)
        vital_items = {
            220045: 'heart_rate',
            220050: 'systolic_bp', 
            220051: 'diastolic_bp',
            220210: 'respiratory_rate',
            223761: 'temperature',
            220277: 'oxygen_saturation'
        }
        
        # Filter to vital signs only
        vitals = chartevents[chartevents['itemid'].isin(vital_items.keys())].copy()
        
        if vitals.empty:
            return pd.DataFrame()
        
        # Map item IDs to readable names
        vitals['vital_type'] = vitals['itemid'].map(vital_items)
        
        # Standardize columns
        vitals = vitals.rename(columns={
            'subject_id': 'patient_id',
            'charttime': 'measurement_date',
            'valuenum': 'value'
        })
        
        # Convert to wide format
        vitals_wide = vitals.pivot_table(
            index=['patient_id', 'measurement_date'],
            columns='vital_type',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        return vitals_wide
    
    def _extract_labs(self, labevents: pd.DataFrame) -> pd.DataFrame:
        """Extract lab values from labevents"""
        # Common lab item IDs in MIMIC-IV
        lab_items = {
            50861: 'hemoglobin',
            50802: 'hematocrit', 
            50931: 'glucose',
            50912: 'creatinine',
            50983: 'sodium',
            50971: 'potassium'
        }
        
        # Filter to common labs
        labs = labevents[labevents['itemid'].isin(lab_items.keys())].copy()
        
        if labs.empty:
            return pd.DataFrame()
        
        # Map item IDs to readable names
        labs['lab_type'] = labs['itemid'].map(lab_items)
        
        # Standardize columns
        labs = labs.rename(columns={
            'subject_id': 'patient_id',
            'charttime': 'lab_date',
            'valuenum': 'value'
        })
        
        # Convert to wide format
        labs_wide = labs.pivot_table(
            index=['patient_id', 'lab_date'],
            columns='lab_type', 
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        return labs_wide
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Return MIMIC-IV specific feature configuration"""
        return {
            'patient_id_column': 'patient_id',
            'date_columns': ['admission_date', 'discharge_date', 'measurement_date', 'lab_date'],
            'vital_features': ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature', 'oxygen_saturation'],
            'lab_features': ['hemoglobin', 'hematocrit', 'glucose', 'creatinine', 'sodium', 'potassium'],
            'outcome_window_days': 90,
            'prediction_window_days': 30
        }

class eICUAdapter(DatasetAdapter):
    """Adapter for eICU dataset"""
    
    def detect_format(self, data_path: str) -> bool:
        """Check if this looks like eICU data"""
        path = Path(data_path)
        
        eicu_files = ['patient.csv', 'vitalPeriodic.csv', 'lab.csv']
        found_files = [f for f in eicu_files if (path / f).exists()]
        
        return len(found_files) >= 2
    
    def load_and_standardize(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load eICU data and standardize"""
        path = Path(data_path)
        datasets = {}
        
        # Load and standardize each table
        if (path / 'patient.csv').exists():
            patients = pd.read_csv(path / 'patient.csv')
            datasets['patients'] = self._standardize_eicu_patients(patients)
        
        if (path / 'vitalPeriodic.csv').exists():
            vitals = pd.read_csv(path / 'vitalPeriodic.csv')
            datasets['vitals'] = self._standardize_eicu_vitals(vitals)
        
        if (path / 'lab.csv').exists():
            labs = pd.read_csv(path / 'lab.csv')
            datasets['labs'] = self._standardize_eicu_labs(labs)
        
        logger.info(f"Loaded eICU dataset with {len(datasets)} tables")
        return datasets
    
    def _standardize_eicu_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize eICU patient data"""
        column_mapping = {
            'patientunitstayid': 'patient_id',
            'age': 'age',
            'gender': 'gender',
            'unitadmittime24': 'admission_date'
        }
        
        return df.rename(columns=column_mapping)
    
    def _standardize_eicu_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize eICU vitals"""
        column_mapping = {
            'patientunitstayid': 'patient_id',
            'observationoffset': 'time_offset',
            'heartrate': 'heart_rate',
            'systemicsystolic': 'systolic_bp',
            'systemicdiastolic': 'diastolic_bp',
            'respiratoryrate': 'respiratory_rate',
            'temperature': 'temperature',
            'sao2': 'oxygen_saturation'
        }
        
        return df.rename(columns=column_mapping)
    
    def _standardize_eicu_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize eICU lab data"""
        column_mapping = {
            'patientunitstayid': 'patient_id',
            'labresultoffset': 'time_offset',
            'labname': 'lab_type',
            'labresult': 'value'
        }
        
        standardized = df.rename(columns=column_mapping)
        
        # Convert to wide format
        labs_wide = standardized.pivot_table(
            index=['patient_id', 'time_offset'],
            columns='lab_type',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        return labs_wide
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Return eICU specific feature configuration"""
        return {
            'patient_id_column': 'patient_id',
            'time_column': 'time_offset',
            'vital_features': ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature', 'oxygen_saturation'],
            'lab_features': ['glucose', 'creatinine', 'hemoglobin', 'sodium', 'potassium'],
            'outcome_window_days': 90,
            'prediction_window_days': 30
        }

class DiabetesAdapter(DatasetAdapter):
    """Adapter for Diabetes 130-US hospitals dataset"""
    
    def detect_format(self, data_path: str) -> bool:
        """Check if this looks like the diabetes dataset"""
        path = Path(data_path)
        
        # Look for diabetic_data.csv or check for readmitted column
        diabetes_file = path / 'diabetic_data.csv'
        if diabetes_file.exists():
            return True
            
        # Check any CSV file for readmitted column
        for csv_file in path.glob('*.csv'):
            try:
                df_sample = pd.read_csv(csv_file, nrows=1)
                if 'readmitted' in df_sample.columns:
                    return True
            except:
                continue
                
        return False
    
    def load_and_standardize(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load diabetes dataset and preprocess into patients.csv and outcomes.csv"""
        path = Path(data_path)
        
        # Find the diabetes data file
        diabetes_file = None
        if (path / 'diabetic_data.csv').exists():
            diabetes_file = path / 'diabetic_data.csv'
        else:
            # Look for any CSV with readmitted column
            for csv_file in path.glob('*.csv'):
                try:
                    df_sample = pd.read_csv(csv_file, nrows=1)
                    if 'readmitted' in df_sample.columns:
                        diabetes_file = csv_file
                        break
                except:
                    continue
        
        if not diabetes_file:
            raise ValueError("No diabetes dataset file found")
        
        logger.info(f"Loading diabetes dataset from: {diabetes_file}")
        df = pd.read_csv(diabetes_file)
        
        # Preprocess the dataset
        datasets = self._preprocess_diabetes_data(df, path)
        
        return datasets
    
    def _preprocess_diabetes_data(self, df: pd.DataFrame, save_path: Path) -> Dict[str, pd.DataFrame]:
        """Preprocess diabetes data into patients.csv and outcomes.csv"""
        
        logger.info(f"Preprocessing diabetes dataset with {len(df)} records")
        
        # Add patient_id column (1..n)
        df = df.reset_index(drop=True)
        df['patient_id'] = (df.index + 1).astype(str)
        
        # Create patients.csv with demographics and clinical features
        patient_columns = [
            'patient_id', 'race', 'gender', 'age', 'weight',
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'payer_code', 'medical_specialty',
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
            'max_glu_serum', 'A1Cresult',
            # Medications
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone', 'change', 'diabetesMed'
        ]
        
        # Filter to available columns
        available_patient_cols = [col for col in patient_columns if col in df.columns]
        patients_df = df[available_patient_cols].copy()
        
        # Create outcomes.csv
        # event_within_90d: 1 if readmitted = '<30' or '>30', else 0
        patients_df['event_within_90d'] = (
            (df['readmitted'] == '<30') | (df['readmitted'] == '>30')
        ).astype(int)
        
        # time_to_event: use discharge_disposition_id as proxy or default 90 if censored
        # For demonstration, we'll create realistic time-to-event data
        def calculate_time_to_event(row):
            if row['readmitted'] == '<30':
                # Readmitted within 30 days - sample from 1-30 days
                return np.random.uniform(1, 30)
            elif row['readmitted'] == '>30':
                # Readmitted after 30 days - sample from 31-90 days
                return np.random.uniform(31, 90)
            else:
                # Not readmitted - censored at 90 days
                return 90.0
        
        np.random.seed(42)  # For reproducibility
        patients_df['time_to_event'] = df.apply(calculate_time_to_event, axis=1)
        
        # Create outcomes dataframe
        outcomes_df = patients_df[['patient_id', 'event_within_90d', 'time_to_event']].copy()
        
        # Remove outcome columns from patients_df
        patients_df = patients_df.drop(['event_within_90d', 'time_to_event'], axis=1)
        
        # Save processed files
        patients_file = save_path / 'patients.csv'
        outcomes_file = save_path / 'outcomes.csv'
        
        patients_df.to_csv(patients_file, index=False)
        outcomes_df.to_csv(outcomes_file, index=False)
        
        logger.info(f"Saved processed patients data to: {patients_file}")
        logger.info(f"Saved outcomes data to: {outcomes_file}")
        logger.info(f"Event rate: {outcomes_df['event_within_90d'].mean():.1%}")
        
        return {
            'patients': patients_df,
            'outcomes': outcomes_df
        }
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Return diabetes dataset specific feature configuration"""
        return {
            'patient_id_column': 'patient_id',
            'outcome_column': 'event_within_90d',
            'time_column': 'time_to_event',
            'demographic_features': ['age', 'gender', 'race', 'weight'],
            'clinical_features': [
                'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses'
            ],
            'lab_features': ['max_glu_serum', 'A1Cresult'],
            'medication_features': [
                'metformin', 'insulin', 'diabetesMed', 'change'
            ],
            'outcome_window_days': 90,
            'prediction_window_days': 30,
            'auto_detect_features': True
        }

class CustomAdapter(DatasetAdapter):
    """Adapter for custom dataset formats"""
    
    def detect_format(self, data_path: str) -> bool:
        """Always returns True as fallback adapter"""
        return True
    
    def load_and_standardize(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load custom format data"""
        path = Path(data_path)
        datasets = {}
        
        # Try to load any CSV files found
        csv_files = list(path.glob('*.csv'))
        
        for csv_file in csv_files:
            table_name = csv_file.stem
            df = pd.read_csv(csv_file)
            datasets[table_name] = df
            logger.info(f"Loaded {table_name} with {len(df)} rows")
        
        # Auto-detect structure
        datasets = self._auto_detect_structure(datasets)
        
        return datasets
    
    def _auto_detect_structure(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Automatically detect and standardize data structure"""
        
        # Try to identify patient table
        patient_indicators = ['patient', 'demographics', 'subjects']
        patients_table = None
        
        for table_name, df in datasets.items():
            if any(indicator in table_name.lower() for indicator in patient_indicators):
                patients_table = table_name
                break
        
        # Try to identify ID columns
        id_indicators = ['id', 'patient', 'subject', 'person']
        
        for table_name, df in datasets.items():
            id_col = None
            for col in df.columns:
                if any(indicator in col.lower() for indicator in id_indicators):
                    id_col = col
                    break
            
            if id_col and id_col != 'patient_id':
                df.rename(columns={id_col: 'patient_id'}, inplace=True)
                logger.info(f"Mapped {id_col} to patient_id in {table_name}")
        
        return datasets
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Return generic feature configuration"""
        return {
            'patient_id_column': 'patient_id',
            'outcome_window_days': 90,
            'prediction_window_days': 30,
            'auto_detect_features': True
        }

class DynamicDataLoader:
    """Main data loader that automatically detects and loads different dataset formats"""
    
    def __init__(self):
        self.adapters = [
            MIMICAdapter(),
            eICUAdapter(),
            DiabetesAdapter(),  # Add diabetes adapter before custom
            CustomAdapter()  # Fallback adapter
        ]
        self.current_adapter = None
        self.feature_config = None
    
    def load_dataset(self, data_path: str, config_override: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Automatically detect dataset format and load data
        
        Args:
            data_path: Path to dataset directory
            config_override: Optional configuration overrides
            
        Returns:
            Dictionary of standardized DataFrames
        """
        
        logger.info(f"Loading dataset from: {data_path}")
        
        # Detect appropriate adapter
        for adapter in self.adapters:
            if adapter.detect_format(data_path):
                self.current_adapter = adapter
                logger.info(f"Using {adapter.__class__.__name__} for data loading")
                break
        
        if not self.current_adapter:
            raise ValueError("No suitable adapter found for dataset format")
        
        # Load and standardize data
        datasets = self.current_adapter.load_and_standardize(data_path)
        
        # Get feature configuration
        self.feature_config = self.current_adapter.get_feature_config()
        
        # Apply any configuration overrides
        if config_override:
            self.feature_config.update(config_override)
        
        # Validate and enhance datasets
        datasets = self._validate_and_enhance(datasets)
        
        return datasets
    
    def _validate_and_enhance(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and enhance loaded datasets"""
        
        # Ensure patient_id is consistent across tables
        patient_id_col = self.feature_config.get('patient_id_column', 'patient_id')
        
        for table_name, df in datasets.items():
            if patient_id_col in df.columns:
                df[patient_id_col] = df[patient_id_col].astype(str)
        
        # Auto-detect features if needed
        if self.feature_config.get('auto_detect_features', False):
            self._auto_detect_features(datasets)
        
        return datasets
    
    def _auto_detect_features(self, datasets: Dict[str, pd.DataFrame]):
        """Automatically detect vital and lab features"""
        
        # Common vital sign patterns
        vital_patterns = ['heart', 'bp', 'pressure', 'temp', 'resp', 'oxygen', 'pulse']
        lab_patterns = ['glucose', 'creatinine', 'sodium', 'potassium', 'hemoglobin', 'hematocrit']
        
        detected_vitals = []
        detected_labs = []
        
        for table_name, df in datasets.items():
            for col in df.columns:
                col_lower = col.lower()
                
                # Check for vital signs
                if any(pattern in col_lower for pattern in vital_patterns):
                    if col not in detected_vitals:
                        detected_vitals.append(col)
                
                # Check for lab values
                if any(pattern in col_lower for pattern in lab_patterns):
                    if col not in detected_labs:
                        detected_labs.append(col)
        
        # Update feature config
        if detected_vitals:
            self.feature_config['vital_features'] = detected_vitals
            logger.info(f"Auto-detected vital features: {detected_vitals}")
        
        if detected_labs:
            self.feature_config['lab_features'] = detected_labs
            logger.info(f"Auto-detected lab features: {detected_labs}")
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Return current feature configuration"""
        return self.feature_config or {}
    
    def create_demo_dataset(self, save_path: str = None) -> Dict[str, pd.DataFrame]:
        """Create a small demo dataset for development/testing"""
        
        # Create minimal demo data
        patients = pd.DataFrame({
            'patient_id': [f'DEMO_{i:03d}' for i in range(1, 51)],
            'age': np.random.randint(40, 90, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'admission_date': pd.date_range('2023-01-01', '2023-12-31', periods=50)
        })
        
        # Create some vitals data
        vitals_data = []
        for _, patient in patients.iterrows():
            for day in range(30):  # 30 days of data
                vitals_data.append({
                    'patient_id': patient['patient_id'],
                    'measurement_date': patient['admission_date'] + pd.Timedelta(days=day),
                    'heart_rate': np.random.normal(80, 15),
                    'systolic_bp': np.random.normal(120, 20),
                    'diastolic_bp': np.random.normal(80, 10),
                    'temperature': np.random.normal(98.6, 1)
                })
        
        vitals = pd.DataFrame(vitals_data)
        
        # Create outcomes (random for demo)
        outcomes = pd.DataFrame({
            'patient_id': patients['patient_id'],
            'deterioration_90d': np.random.choice([0, 1], 50, p=[0.7, 0.3])
        })
        
        demo_dataset = {
            'patients': patients,
            'vitals': vitals,
            'outcomes': outcomes
        }
        
        # Save if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            for name, df in demo_dataset.items():
                df.to_csv(Path(save_path) / f"{name}.csv", index=False)
            logger.info(f"Demo dataset saved to {save_path}")
        
        # Set up basic feature config
        self.feature_config = {
            'patient_id_column': 'patient_id',
            'vital_features': ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature'],
            'lab_features': [],
            'outcome_column': 'deterioration_90d',
            'outcome_window_days': 90,
            'prediction_window_days': 30
        }
        
        return demo_dataset

if __name__ == "__main__":
    # Test the dynamic loader
    loader = DynamicDataLoader()
    
    # Create demo dataset for testing
    demo_data = loader.create_demo_dataset("../data/demo")
    
    print("Demo dataset created successfully!")
    print("Feature configuration:", loader.get_feature_config())