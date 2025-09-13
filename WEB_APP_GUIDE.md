# 🌐 MDM2 Inhibition Predictor - Web Application Guide

## 🚀 Quick Start (For Non-Coders!)

### Step 1: Launch the Web App
```bash
# Option 1: Easy launch (recommended)
python3 start_web_app.py

# Option 2: Manual launch
streamlit run app.py
```

### Step 2: Open Your Web Browser
- The app will automatically open at: **http://localhost:8501**
- If it doesn't open automatically, copy that link into your browser

### Step 3: Start Predicting!
You're ready to use the MDM2 prediction tool! 🎉

---

## 📱 How to Use the Web Interface

### 🔬 **Input Your Molecules**

**Option 1: Single Molecule**
- Enter one SMILES string (e.g., `CCO` for ethanol)
- Click "Predict MDM2 Inhibition"

**Option 2: Multiple Molecules**
- Enter multiple SMILES, one per line
- Great for testing several compounds at once

**Option 3: Upload CSV File**
- Create a CSV file with a "smiles" column
- Upload it to process hundreds of molecules

### 📊 **Understanding Results**

For each molecule, you'll see:
- **🔴 LIKELY INHIBITS MDM2** = This molecule probably blocks MDM2
- **🟢 UNLIKELY TO INHIBIT** = This molecule probably doesn't affect MDM2  
- **Probability**: How confident the AI is (0-100%)
- **Confidence**: How reliable this prediction is

### 💾 **Download Results**
- For multiple predictions, click "Download Results as CSV"
- Save your results for further analysis

---

## 🧬 What Are SMILES?

**SMILES** = **S**implified **M**olecular **I**nput **L**ine **E**ntry **S**ystem

It's a way to represent molecules as text strings:
- `CCO` = Ethanol (drinking alcohol)
- `CC(=O)O` = Acetic acid (vinegar)  
- `c1ccccc1` = Benzene ring
- `CN1CCC[C@H]1C2=CN=CC=C2` = Nicotine

**Where to get SMILES?**
- PubChem database: https://pubchem.ncbi.nlm.nih.gov/
- ChemSpider: http://chemspider.com/
- Draw molecules online: https://web.chemdoodle.com/

---

## 🎯 What is MDM2?

**MDM2** (Mouse Double Minute 2) is a protein that acts as a "brake" on p53, which is known as the "guardian of the genome."

**Why it matters for cancer:**
- p53 normally stops cancer by killing damaged cells
- MDM2 can turn off p53, allowing cancer to grow
- **MDM2 inhibitors** can restore p53 function = potential cancer treatment

**Our AI Model Predicts:**
- Whether your molecule will block (inhibit) MDM2
- How confident we are in that prediction
- This helps identify potential anti-cancer drugs

---

## 🛠️ Technical Features

### 🤖 **Advanced AI Architecture**
- **Graph Neural Networks**: Understands molecular structure
- **Adversarial Training**: Makes predictions more robust
- **Evolutionary Optimization**: Automatically finds best features
- **Confidence Scoring**: Tells you how reliable each prediction is

### 📈 **Training Data**
- Trained on ChEMBL database
- 50+ validated MDM2 inhibitors and non-inhibitors
- Professional pharmaceutical data

### ⚡ **Performance**
- Instant predictions (seconds)
- Batch processing (hundreds of molecules)
- High accuracy with confidence scores
- User-friendly web interface

---

## 🔧 Installation (For Technical Users)

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install streamlit
```

### Quick Setup
```bash
# Clone/download the project
cd p53mdm2/

# Install dependencies  
pip install -r requirements.txt
pip install streamlit

# Launch web app
python3 start_web_app.py
```

### Manual Launch
```bash
streamlit run app.py
```

---

## ⚠️ Important Disclaimers

### 🔬 **Research Use Only**
- This tool is for **research purposes only**
- Not for medical diagnosis or treatment
- Always validate predictions experimentally

### 🧪 **Limitations**
- Predictions are computational estimates
- Real biological activity may differ
- Some complex molecules may not be processed correctly
- Always consult medicinal chemistry experts

### 📊 **Data Quality**
- Model trained on limited dataset (50 compounds for demo)
- Production version would use larger datasets (800+ compounds)
- Confidence scores indicate prediction reliability

---

## 🆘 Troubleshooting

### **App Won't Start**
```bash
# Install missing dependencies
pip install streamlit pandas numpy torch scikit-learn

# Try manual launch
streamlit run app.py
```

### **Browser Won't Open**
- Manually go to: http://localhost:8501
- Try a different browser
- Check if port 8501 is blocked

### **Prediction Errors**
- Check SMILES format (no spaces, valid chemistry)
- Try simpler molecules first
- Some complex structures may fail

### **Performance Issues**
- Use smaller batches (< 100 molecules at once)
- Close other applications to free memory
- Try individual predictions instead of batch

---

## 📞 Support & Contact

**Developer**: Chetanya Pandey  
**Project**: P53-MDM2 Drug Design Prediction Model  
**Version**: 1.0  
**Date**: September 2025

**For Support:**
- Check this guide first
- Try the example SMILES provided
- Restart the application if needed

---

## 🎉 **Ready to Discover New Cancer Drugs?**

Your MDM2 inhibition prediction tool is ready to use! 

🔬 **Start with simple molecules**  
📈 **Build confidence with the predictions**  
🚀 **Scale up to your research compounds**

**Happy drug discovery!** 🧬✨