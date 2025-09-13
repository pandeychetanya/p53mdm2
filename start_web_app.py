"""
Simple script to launch the MDM2 prediction web app
"""

import subprocess
import sys
import webbrowser
import time

def main():
    """Launch the Streamlit web application"""
    
    print("🧬 Starting MDM2 Inhibition Prediction Web App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    print("\n🚀 Launching web application...")
    print("📱 The app will open in your default web browser")
    print("🌐 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n💡 To stop the app, press Ctrl+C in this terminal")
    print("=" * 50)
    
    # Wait a moment then open browser
    time.sleep(2)
    webbrowser.open('http://localhost:8501')
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web app...")
        print("✅ Web app stopped successfully")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("💡 Try running manually: streamlit run app.py")

if __name__ == "__main__":
    main()