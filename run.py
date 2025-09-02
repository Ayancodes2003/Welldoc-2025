"""
AI Risk Prediction Engine - Easy Startup Script

This script provides multiple options for running the dashboard.
"""
import subprocess
import sys
import os
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("=" * 60)
    print("🏥 AI RISK PREDICTION ENGINE")
    print("=" * 60)
    print("Welcome! Choose how you'd like to run the dashboard:")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False

def run_system_test():
    """Run the system test"""
    print("🔧 Running system tests...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests passed!")
            return True
        else:
            print("❌ System tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_streamlit():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Streamlit dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def run_demo_script():
    """Run the demo walkthrough script"""
    print("📋 Starting demo walkthrough...")
    print("This will guide you through the dashboard features")
    print()
    
    try:
        subprocess.run([sys.executable, "demo_script.py"])
    except Exception as e:
        print(f"❌ Error running demo script: {e}")

def main():
    """Main startup interface"""
    print_header()
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        print("\nSelect an option:")
        print("1. 🚀 Run Dashboard (Recommended)")
        print("2. 🔧 Run System Test First")
        print("3. 📋 Demo Walkthrough Guide")
        print("4. ℹ️  Show Installation Instructions")
        print("5. 🐳 Docker Instructions")
        print("6. ❌ Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_streamlit()
        
        elif choice == "2":
            if run_system_test():
                print("\n✅ Tests passed! Ready to run dashboard.")
                input("Press Enter to start dashboard...")
                run_streamlit()
            else:
                print("\n⚠️  Some tests failed. Check error messages above.")
        
        elif choice == "3":
            run_demo_script()
        
        elif choice == "4":
            show_installation_instructions()
        
        elif choice == "5":
            show_docker_instructions()
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def show_installation_instructions():
    """Show installation instructions"""
    print("\n📦 INSTALLATION INSTRUCTIONS")
    print("=" * 40)
    print()
    print("1. Clone the repository:")
    print("   git clone <repository-url>")
    print("   cd Welldoc-2025")
    print()
    print("2. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Run the dashboard:")
    print("   streamlit run app.py")
    print()
    print("4. Open in browser:")
    print("   http://localhost:8501")
    print()

def show_docker_instructions():
    """Show Docker instructions"""
    print("\n🐳 DOCKER INSTRUCTIONS")
    print("=" * 30)
    print()
    print("1. Build the container:")
    print("   docker build -t risk-prediction-dashboard .")
    print()
    print("2. Run the container:")
    print("   docker run -p 8501:8501 risk-prediction-dashboard")
    print()
    print("3. Open in browser:")
    print("   http://localhost:8501")
    print()
    print("Note: Docker must be installed and running on your system.")
    print()

if __name__ == "__main__":
    main()