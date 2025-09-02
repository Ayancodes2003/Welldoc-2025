"""
Demo Script for AI Risk Prediction Engine

This script provides a guided walkthrough of the dashboard functionality.
Run this alongside the Streamlit dashboard for a comprehensive demo.
"""
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def print_demo_header():
    """Print demo introduction"""
    print("=" * 70)
    print("🏥 AI RISK PREDICTION ENGINE - DEMO WALKTHROUGH")
    print("=" * 70)
    print()
    print("Welcome to the AI Risk Prediction Engine demonstration!")
    print("This dashboard predicts 90-day deterioration risk for chronic care patients.")
    print()
    print("🎯 Key Features:")
    print("  • Dynamic data pipeline (MIMIC-IV, eICU, custom formats)")
    print("  • Baseline ML models (Logistic Regression, Random Forest)")  
    print("  • SHAP explanations in plain English")
    print("  • Interactive dashboard with 3 main views")
    print("  • Clinical recommendations")
    print()

def demo_step(step_num: int, title: str, description: str, pause: bool = True):
    """Print a demo step"""
    print(f"\n{'=' * 15} STEP {step_num}: {title.upper()} {'=' * 15}")
    print(description)
    
    if pause:
        input("\nPress Enter to continue...")

def main():
    """Run the complete demo walkthrough"""
    
    print_demo_header()
    
    # Step 1: Overview
    demo_step(1, "Dashboard Overview", 
    """
    The dashboard has three main views accessible via the sidebar:
    
    👥 COHORT OVERVIEW:
    • See all patients and their risk scores
    • Filter and sort by risk levels
    • View risk distribution across the population
    • Quick patient selection for detailed analysis
    
    👤 PATIENT DETAIL:
    • Individual risk assessment with explanations
    • Top 5 risk factors in plain English
    • Patient timeline with vital signs
    • Personalized clinical recommendations
    
    ⚙️ ADMIN DASHBOARD:
    • Model performance metrics (AUC-ROC, etc.)
    • Global feature importance
    • Data quality overview
    • Performance visualizations
    """)
    
    # Step 2: Demo Mode
    demo_step(2, "Demo Mode Setup",
    """
    📋 CURRENT SETUP:
    • Demo Mode is ON (using synthetic data)
    • 50 synthetic patients with realistic profiles
    • Age-correlated risk scores
    • All features working without real data
    
    🔄 TO USE YOUR OWN DATA:
    1. Turn off "Demo Mode" in sidebar
    2. Place your data files in the 'data/' directory
    3. System auto-detects format (MIMIC-IV, eICU, custom)
    4. Dashboard adapts automatically
    """)
    
    # Step 3: Cohort View Demo
    demo_step(3, "Cohort View Walkthrough",
    """
    Navigate to "Cohort Overview" in the sidebar:
    
    📊 WHAT YOU'LL SEE:
    • Summary metrics (Total, High/Medium/Low risk patients)
    • Risk score distribution histogram
    • Risk category pie chart
    • Searchable patient table with:
      - Patient ID, Risk Score, Demographics
      - Risk trends and top risk factors
      - Sortable columns
    
    🎯 TRY THIS:
    1. Filter by "High Risk (>70%)" 
    2. Sort by "Risk Score (High to Low)"
    3. Search for a specific patient ID
    4. Click on a patient row and "View Patient Detail"
    """)
    
    # Step 4: Patient Detail Demo
    demo_step(4, "Patient Detail Walkthrough",
    """
    Select a patient and navigate to "Patient Detail":
    
    👤 PATIENT PROFILE:
    • Demographics (Age, Gender, Admissions)
    • Recent vital signs
    • Risk gauge visualization (color-coded)
    
    🔍 RISK ANALYSIS:
    • Risk score percentage and category
    • Top 5 risk factors with impact levels
    • Plain English explanations like:
      "Average heart rate of 95 bpm increases risk"
      "Patient is over 75 years old increases risk"
    
    📊 PATIENT TIMELINE:
    • Vital signs over time
    • Risk trajectory (when available)
    • Trend analysis
    
    📋 RECOMMENDATIONS:
    • Personalized clinical suggestions
    • Based on specific risk factors
    • Actionable next steps
    """)
    
    # Step 5: Admin Dashboard Demo
    demo_step(5, "Admin Dashboard Walkthrough",
    """
    Navigate to "Admin Dashboard":
    
    ⚙️ MODEL INFORMATION:
    • Model type and training status
    • Demo vs Production mode
    • Feature count and sample size
    
    📈 PERFORMANCE METRICS:
    • AUC-ROC (Area Under ROC Curve)
    • AUC-PR (Area Under Precision-Recall)
    • Accuracy and other metrics
    • Performance visualization plots
    
    🌍 GLOBAL EXPLANATIONS:
    • Most important features across all patients
    • Feature importance percentages
    • Model behavior summary
    
    📊 DATA QUALITY:
    • Dataset summaries
    • Missing data percentages
    • Feature configuration details
    """)
    
    # Step 6: Key Use Cases
    demo_step(6, "Key Use Cases",
    """
    🏥 CLINICAL USE CASES:
    
    1. POPULATION HEALTH MONITORING:
       • Identify high-risk patients for proactive care
       • Resource allocation based on risk distribution
       • Track risk trends over time
    
    2. INDIVIDUAL PATIENT CARE:
       • Understand specific risk factors
       • Get actionable recommendations
       • Monitor patient trajectory
    
    3. QUALITY IMPROVEMENT:
       • Analyze model performance
       • Identify most impactful risk factors
       • Data quality monitoring
    
    4. RESEARCH & ANALYTICS:
       • Global feature importance analysis
       • Population risk patterns
       • Model validation and improvement
    """)
    
    # Step 7: Technical Highlights
    demo_step(7, "Technical Highlights",
    """
    🛠️ TECHNICAL FEATURES:
    
    • DYNAMIC DATA PIPELINE:
      - Auto-detects MIMIC-IV, eICU, custom formats
      - Automatic feature engineering
      - Handles missing data gracefully
    
    • MACHINE LEARNING:
      - Baseline models (Logistic Regression, Random Forest)
      - Automatic scaling and encoding
      - Cross-validation for robust evaluation
    
    • EXPLAINABLE AI:
      - SHAP values for transparent predictions
      - Plain-English risk factor explanations
      - Global and local interpretability
    
    • PRODUCTION READY:
      - Docker containerization
      - Streamlit caching for performance
      - Modular architecture for extensibility
    """)
    
    # Step 8: Next Steps
    demo_step(8, "Next Steps & Extensions",
    """
    🚀 FUTURE ENHANCEMENTS:
    
    • Time-to-event modeling (survival analysis)
    • Real-time data streaming capabilities
    • Advanced transformer models
    • Multi-outcome predictions
    • EHR system integration
    • Automated model retraining
    
    🔧 CUSTOMIZATION OPTIONS:
    
    • Add new data source adapters
    • Customize feature engineering
    • Modify risk thresholds
    • Enhance explanation templates
    • Add new visualization types
    
    📋 DEPLOYMENT OPTIONS:
    
    • Local installation (pip install)
    • Docker container deployment
    • Cloud platform hosting
    • Integration with existing systems
    """, pause=False)
    
    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Thank you for exploring the AI Risk Prediction Engine!")
    print("For questions or support, please refer to the README.md file.")
    print()
    print("The dashboard is running at: http://localhost:8501")
    print()

if __name__ == "__main__":
    main()