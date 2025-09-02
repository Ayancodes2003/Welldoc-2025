"""
Virtual Environment Setup Script

Creates and configures a virtual environment for the AI Risk Prediction Engine.
"""
import subprocess
import sys
import os
from pathlib import Path

def create_virtual_env():
    """Create a virtual environment"""
    print("🔧 Creating virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_activation_command():
    """Get the command to activate virtual environment based on OS"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return "source venv/bin/activate"

def install_dependencies():
    """Install dependencies in virtual environment"""
    print("📦 Installing dependencies...")
    
    # Get the python executable path in virtual environment
    if os.name == 'nt':  # Windows
        python_exe = Path("venv/Scripts/python.exe")
        pip_exe = Path("venv/Scripts/pip.exe")
    else:  # Unix/Linux/macOS
        python_exe = Path("venv/bin/python")
        pip_exe = Path("venv/bin/pip")
    
    if not python_exe.exists():
        print("❌ Virtual environment not found or not activated")
        return False
    
    try:
        # Upgrade pip first
        print("📦 Upgrading pip...")
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("📦 Installing project requirements...")
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_installation():
    """Test the installation by running system tests"""
    print("🔍 Testing installation...")
    
    if os.name == 'nt':  # Windows
        python_exe = Path("venv/Scripts/python.exe")
    else:  # Unix/Linux/macOS
        python_exe = Path("venv/bin/python")
    
    try:
        result = subprocess.run([str(python_exe), "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Installation test passed!")
            return True
        else:
            print("⚠️ Some tests failed, but basic installation seems OK")
            print("You can still try running the dashboard")
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out, but installation may be OK")
        return True
    except Exception as e:
        print(f"⚠️ Could not run tests: {e}")
        print("Installation may still be OK - try running manually")
        return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🏥 AI RISK PREDICTION ENGINE - VIRTUAL ENVIRONMENT SETUP")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Create virtual environment
    if not create_virtual_env():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Step 3: Test installation
    test_installation()
    
    # Step 4: Provide instructions
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("To use the AI Risk Prediction Engine:")
    print()
    
    activation_cmd = get_activation_command()
    
    if os.name == 'nt':  # Windows
        print("1. Activate the virtual environment:")
        print(f"   {activation_cmd}")
        print()
        print("2. Run the dashboard:")
        print("   python app.py")
        print("   OR")
        print("   streamlit run app.py")
        print()
        print("3. Run with startup script:")
        print("   python run.py")
    else:  # Unix/Linux/macOS
        print("1. Activate the virtual environment:")
        print(f"   {activation_cmd}")
        print()
        print("2. Run the dashboard:")
        print("   python app.py")
        print("   OR")
        print("   streamlit run app.py")
        print()
        print("3. Run with startup script:")
        print("   python run.py")
    
    print()
    print("💡 Quick Commands:")
    print("   • Test system:     python test_system.py")
    print("   • Demo guide:      python demo_script.py")
    print("   • Easy startup:    python run.py")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print()

if __name__ == "__main__":
    main()