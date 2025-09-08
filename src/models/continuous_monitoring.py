"""
Continuous Monitoring Feature Engineering

Extends preprocessing to support 30-180 day input windows with rolling features,
temporal analysis, and irregular sampling handling.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ContinuousMonitoringProcessor:
    """Enhanced feature engineering for continuous monitoring scenarios"""
    
    def __init__(self, monitoring_windows: List[int] = [30, 60, 90, 180]):
        """
        Initialize with monitoring windows in days
        
        Args:
            monitoring_windows: List of days to create rolling windows for
        """
        self.monitoring_windows = monitoring_windows
        self.feature_metadata = {}
        
    def generate_continuous_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate comprehensive continuous monitoring features
        
        Args:
            datasets: Dictionary containing patient data tables
            
        Returns:
            Feature matrix with continuous monitoring features
        """
        logger.info("Generating continuous monitoring features...")
        
        # Extract base patient information
        if 'patients' not in datasets:
            raise ValueError("No patients table found in datasets")
            
        patients_df = datasets['patients'].copy()
        
        # Initialize feature matrix
        feature_matrix = patients_df[['patient_id']].copy()
        
        # Add demographic features
        demographic_features = self._extract_demographic_features(patients_df)
        feature_matrix = pd.merge(feature_matrix, demographic_features, on='patient_id', how='left')
        
        # Add temporal vitals features if available
        if 'vitals' in datasets:
            vitals_features = self._extract_temporal_vitals(datasets['vitals'])
            feature_matrix = pd.merge(feature_matrix, vitals_features, on='patient_id', how='left')
            
        # Add temporal labs features if available
        if 'labs' in datasets:
            labs_features = self._extract_temporal_labs(datasets['labs'])
            feature_matrix = pd.merge(feature_matrix, labs_features, on='patient_id', how='left')
            
        # Add clinical complexity features
        clinical_features = self._extract_clinical_complexity(patients_df)
        feature_matrix = pd.merge(feature_matrix, clinical_features, on='patient_id', how='left')
        
        # Add lifestyle and adherence features (synthetic for demonstration)
        lifestyle_features = self._generate_lifestyle_features(feature_matrix)
        feature_matrix = pd.merge(feature_matrix, lifestyle_features, on='patient_id', how='left')
        
        # Fill missing values and finalize
        feature_matrix = self._finalize_features(feature_matrix)
        
        logger.info(f"Generated {len(feature_matrix.columns)-1} features for {len(feature_matrix)} patients")
        return feature_matrix
        
    def _extract_demographic_features(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode demographic features"""
        
        demo_features = patients_df[['patient_id']].copy()
        
        # Age processing
        if 'age' in patients_df.columns:
            # Handle age ranges like '[40-50)' or numeric ages
            def process_age(age_val):
                if pd.isna(age_val):
                    return np.nan
                if isinstance(age_val, str) and '[' in age_val:
                    # Extract midpoint from range like '[40-50)'
                    age_range = age_val.replace('[', '').replace(')', '').replace('-', ',')
                    try:
                        start, end = map(int, age_range.split(','))
                        return (start + end) / 2
                    except:
                        return np.nan
                try:
                    return float(age_val)
                except:
                    return np.nan
            
            demo_features['age'] = patients_df['age'].apply(process_age)
            demo_features['age_over_65'] = (demo_features['age'] >= 65).astype(int)
            demo_features['age_over_75'] = (demo_features['age'] >= 75).astype(int)
            demo_features['age_squared'] = demo_features['age'] ** 2
            
        # Gender encoding
        if 'gender' in patients_df.columns:
            demo_features['gender_encoded'] = (patients_df['gender'] == 'Male').astype(int)
            
        # Race encoding (simplified)
        if 'race' in patients_df.columns:
            race_mapping = {
                'Caucasian': 0, 'African American': 1, 'Hispanic': 2, 
                'Asian': 3, 'Other': 4
            }
            demo_features['race_encoded'] = patients_df['race'].map(race_mapping).fillna(4)
            
        return demo_features
        
    def _extract_temporal_vitals(self, vitals_df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal vital signs features with rolling windows"""
        
        vitals_features = vitals_df[['patient_id']].drop_duplicates().copy()
        
        # Identify vital sign columns
        vital_columns = [col for col in vitals_df.columns 
                        if col not in ['patient_id', 'measurement_date', 'time_offset']]
        
        if not vital_columns:
            return vitals_features
            
        # Prepare temporal data
        vitals_temporal = vitals_df.copy()
        
        # Convert date column if available
        date_col = 'measurement_date' if 'measurement_date' in vitals_df.columns else 'charttime'
        if date_col in vitals_df.columns:
            vitals_temporal[date_col] = pd.to_datetime(vitals_temporal[date_col], errors='coerce')
            vitals_temporal = vitals_temporal.sort_values(['patient_id', date_col])
            
        # Generate features for each monitoring window
        for window_days in self.monitoring_windows:
            window_features = self._calculate_window_vitals(vitals_temporal, vital_columns, window_days)
            vitals_features = pd.merge(vitals_features, window_features, on='patient_id', how='left')
            
        # Add Δt (time since last observation) features
        delta_features = self._calculate_delta_t_vitals(vitals_temporal, vital_columns)
        vitals_features = pd.merge(vitals_features, delta_features, on='patient_id', how='left')
        
        return vitals_features
        
    def _extract_temporal_labs(self, labs_df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal laboratory features with rolling windows"""
        
        labs_features = labs_df[['patient_id']].drop_duplicates().copy()
        
        # Identify lab columns
        lab_columns = [col for col in labs_df.columns 
                      if col not in ['patient_id', 'lab_date', 'time_offset']]
        
        if not lab_columns:
            return labs_features
            
        # Prepare temporal data
        labs_temporal = labs_df.copy()
        
        # Convert date column if available
        date_col = 'lab_date' if 'lab_date' in labs_df.columns else 'charttime'
        if date_col in labs_df.columns:
            labs_temporal[date_col] = pd.to_datetime(labs_temporal[date_col], errors='coerce')
            labs_temporal = labs_temporal.sort_values(['patient_id', date_col])
            
        # Generate features for each monitoring window
        for window_days in self.monitoring_windows:
            window_features = self._calculate_window_labs(labs_temporal, lab_columns, window_days)
            labs_features = pd.merge(labs_features, window_features, on='patient_id', how='left')
            
        # Add Δt features for labs
        delta_features = self._calculate_delta_t_labs(labs_temporal, lab_columns)
        labs_features = pd.merge(labs_features, delta_features, on='patient_id', how='left')
        
        return labs_features
        
    def _calculate_window_vitals(self, vitals_df: pd.DataFrame, vital_columns: List[str], 
                                window_days: int) -> pd.DataFrame:
        """Calculate rolling window statistics for vitals"""
        
        window_features = vitals_df[['patient_id']].drop_duplicates().copy()
        
        for vital in vital_columns:
            if vital not in vitals_df.columns:
                continue
                
            # Calculate window statistics per patient
            vital_stats = []
            
            for patient_id in vitals_df['patient_id'].unique():
                patient_vitals = vitals_df[vitals_df['patient_id'] == patient_id]
                
                # Get most recent window_days of data
                if 'measurement_date' in patient_vitals.columns:
                    # Use most recent measurements within window
                    recent_data = patient_vitals.tail(window_days)
                else:
                    # Use all available data
                    recent_data = patient_vitals
                    
                vital_values = recent_data[vital].dropna()
                
                if len(vital_values) > 0:
                    stats = {
                        'patient_id': patient_id,
                        f'{vital}_mean_{window_days}d': vital_values.mean(),
                        f'{vital}_std_{window_days}d': vital_values.std() if len(vital_values) > 1 else 0,
                        f'{vital}_min_{window_days}d': vital_values.min(),
                        f'{vital}_max_{window_days}d': vital_values.max(),
                        f'{vital}_last_{window_days}d': vital_values.iloc[-1],
                        f'{vital}_slope_{window_days}d': self._calculate_slope(vital_values),
                        f'{vital}_count_{window_days}d': len(vital_values)
                    }
                else:
                    stats = {
                        'patient_id': patient_id,
                        f'{vital}_mean_{window_days}d': np.nan,
                        f'{vital}_std_{window_days}d': np.nan,
                        f'{vital}_min_{window_days}d': np.nan,
                        f'{vital}_max_{window_days}d': np.nan,
                        f'{vital}_last_{window_days}d': np.nan,
                        f'{vital}_slope_{window_days}d': np.nan,
                        f'{vital}_count_{window_days}d': 0
                    }
                    
                vital_stats.append(stats)
                
            # Merge statistics
            vital_stats_df = pd.DataFrame(vital_stats)
            window_features = pd.merge(window_features, vital_stats_df, on='patient_id', how='left')
            
        return window_features
        
    def _calculate_window_labs(self, labs_df: pd.DataFrame, lab_columns: List[str], 
                              window_days: int) -> pd.DataFrame:
        """Calculate rolling window statistics for labs"""
        
        window_features = labs_df[['patient_id']].drop_duplicates().copy()
        
        for lab in lab_columns:
            if lab not in labs_df.columns:
                continue
                
            lab_stats = []
            
            for patient_id in labs_df['patient_id'].unique():
                patient_labs = labs_df[labs_df['patient_id'] == patient_id]
                
                # Get recent window data
                if 'lab_date' in patient_labs.columns:
                    recent_data = patient_labs.tail(window_days // 7)  # Assume weekly labs
                else:
                    recent_data = patient_labs
                    
                lab_values = recent_data[lab].dropna()
                
                if len(lab_values) > 0:
                    stats = {
                        'patient_id': patient_id,
                        f'{lab}_mean_{window_days}d': lab_values.mean(),
                        f'{lab}_last_{window_days}d': lab_values.iloc[-1],
                        f'{lab}_trend_{window_days}d': self._calculate_slope(lab_values),
                        f'{lab}_count_{window_days}d': len(lab_values)
                    }
                else:
                    stats = {
                        'patient_id': patient_id,
                        f'{lab}_mean_{window_days}d': np.nan,
                        f'{lab}_last_{window_days}d': np.nan,
                        f'{lab}_trend_{window_days}d': np.nan,
                        f'{lab}_count_{window_days}d': 0
                    }
                    
                lab_stats.append(stats)
                
            lab_stats_df = pd.DataFrame(lab_stats)
            window_features = pd.merge(window_features, lab_stats_df, on='patient_id', how='left')
            
        return window_features
        
    def _calculate_delta_t_vitals(self, vitals_df: pd.DataFrame, vital_columns: List[str]) -> pd.DataFrame:
        """Calculate time since last observation features"""
        
        delta_features = vitals_df[['patient_id']].drop_duplicates().copy()
        
        for vital in vital_columns:
            if vital not in vitals_df.columns:
                continue
                
            # Calculate time since last valid measurement
            vital_delta = []
            
            for patient_id in vitals_df['patient_id'].unique():
                patient_data = vitals_df[vitals_df['patient_id'] == patient_id]
                valid_measurements = patient_data[patient_data[vital].notna()]
                
                if len(valid_measurements) > 0:
                    # Calculate average and max intervals
                    if 'measurement_date' in valid_measurements.columns:
                        dates = pd.to_datetime(valid_measurements['measurement_date'])
                        if len(dates) > 1:
                            intervals = dates.diff().dt.total_seconds() / 3600  # Hours
                            avg_interval = intervals.mean()
                            max_interval = intervals.max()
                        else:
                            avg_interval = max_interval = 24  # Default 24 hours
                    else:
                        avg_interval = max_interval = 24
                        
                    delta_stats = {
                        'patient_id': patient_id,
                        f'{vital}_avg_interval_hours': avg_interval,
                        f'{vital}_max_interval_hours': max_interval,
                        f'{vital}_measurement_frequency': len(valid_measurements) / 30  # Per 30 days
                    }
                else:
                    delta_stats = {
                        'patient_id': patient_id,
                        f'{vital}_avg_interval_hours': np.nan,
                        f'{vital}_max_interval_hours': np.nan,
                        f'{vital}_measurement_frequency': 0
                    }
                    
                vital_delta.append(delta_stats)
                
            vital_delta_df = pd.DataFrame(vital_delta)
            delta_features = pd.merge(delta_features, vital_delta_df, on='patient_id', how='left')
            
        return delta_features
        
    def _calculate_delta_t_labs(self, labs_df: pd.DataFrame, lab_columns: List[str]) -> pd.DataFrame:
        """Calculate time since last lab observation features"""
        
        delta_features = labs_df[['patient_id']].drop_duplicates().copy()
        
        for lab in lab_columns:
            if lab not in labs_df.columns:
                continue
                
            lab_delta = []
            
            for patient_id in labs_df['patient_id'].unique():
                patient_data = labs_df[labs_df['patient_id'] == patient_id]
                valid_labs = patient_data[patient_data[lab].notna()]
                
                if len(valid_labs) > 0:
                    delta_stats = {
                        'patient_id': patient_id,
                        f'{lab}_days_since_last': np.random.uniform(1, 30),  # Mock data
                        f'{lab}_frequency_per_month': len(valid_labs) / 3  # Assume 3 months
                    }
                else:
                    delta_stats = {
                        'patient_id': patient_id,
                        f'{lab}_days_since_last': np.nan,
                        f'{lab}_frequency_per_month': 0
                    }
                    
                lab_delta.append(delta_stats)
                
            lab_delta_df = pd.DataFrame(lab_delta)
            delta_features = pd.merge(delta_features, lab_delta_df, on='patient_id', how='left')
            
        return delta_features
        
    def _extract_clinical_complexity(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Extract clinical complexity features"""
        
        complexity_features = patients_df[['patient_id']].copy()
        
        # Basic clinical metrics
        clinical_columns = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses'
        ]
        
        for col in clinical_columns:
            if col in patients_df.columns:
                complexity_features[col] = patients_df[col]
                
        # Derived complexity features
        if 'num_medications' in complexity_features.columns:
            complexity_features['polypharmacy'] = (complexity_features['num_medications'] >= 5).astype(int)
            
        if 'number_emergency' in complexity_features.columns:
            complexity_features['high_emergency_use'] = (complexity_features['number_emergency'] >= 2).astype(int)
            
        if 'time_in_hospital' in complexity_features.columns:
            complexity_features['long_stay'] = (complexity_features['time_in_hospital'] >= 7).astype(int)
            
        return complexity_features
        
    def _generate_lifestyle_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic lifestyle and medication adherence features"""
        
        lifestyle_features = feature_matrix[['patient_id']].copy()
        n_patients = len(lifestyle_features)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Exercise habits (0: sedentary, 1: light, 2: moderate, 3: vigorous)
        lifestyle_features['exercise_level'] = np.random.choice([0, 1, 2, 3], n_patients, p=[0.3, 0.4, 0.2, 0.1])
        
        # Diet quality (0-10 scale)
        lifestyle_features['diet_quality_score'] = np.random.normal(6, 2, n_patients).clip(0, 10)
        
        # Smoking status (0: never, 1: former, 2: current)
        lifestyle_features['smoking_status'] = np.random.choice([0, 1, 2], n_patients, p=[0.5, 0.3, 0.2])
        
        # Alcohol consumption (drinks per week)
        lifestyle_features['alcohol_weekly'] = np.random.exponential(3, n_patients).clip(0, 20)
        
        # Sleep quality (hours per night)
        lifestyle_features['sleep_hours'] = np.random.normal(7, 1.5, n_patients).clip(3, 12)
        
        # Medication adherence (0-100%)
        lifestyle_features['med_adherence_percent'] = np.random.beta(3, 1, n_patients) * 100
        
        # Adherence binary flags
        lifestyle_features['med_adherent'] = (lifestyle_features['med_adherence_percent'] >= 80).astype(int)
        lifestyle_features['exercise_adequate'] = (lifestyle_features['exercise_level'] >= 2).astype(int)
        lifestyle_features['non_smoker'] = (lifestyle_features['smoking_status'] == 0).astype(int)
        
        return lifestyle_features
        
    def _calculate_slope(self, values: pd.Series) -> float:
        """Calculate slope of trend line for a series of values"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
            
    def _finalize_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Finalize feature matrix with imputation and scaling"""
        
        # Fill missing values
        numeric_columns = feature_matrix.select_dtypes(include=[np.number]).columns
        feature_matrix[numeric_columns] = feature_matrix[numeric_columns].fillna(
            feature_matrix[numeric_columns].median()
        )
        
        # Record feature metadata
        self.feature_metadata = {
            'total_features': len(feature_matrix.columns) - 1,  # Exclude patient_id
            'demographic_features': len([col for col in feature_matrix.columns if 'age' in col or 'gender' in col or 'race' in col]),
            'temporal_features': len([col for col in feature_matrix.columns if any(str(w) in col for w in self.monitoring_windows)]),
            'lifestyle_features': len([col for col in feature_matrix.columns if any(x in col for x in ['exercise', 'diet', 'smoking', 'alcohol', 'sleep', 'adherence'])]),
            'monitoring_windows': self.monitoring_windows
        }
        
        logger.info(f"Feature engineering complete: {self.feature_metadata}")
        
        return feature_matrix
        
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return feature engineering metadata"""
        return self.feature_metadata