"""
Quick test for ChEMBL data collection with limited compounds
"""

from chembl_collector import ChEMBLDataCollector

if __name__ == "__main__":
    collector = ChEMBLDataCollector()
    
    # Collect smaller dataset for testing (50 compounds)
    print("Fetching 50 MDM2 compounds for testing...")
    mdm2_data = collector.get_mdm2_bioactivity_data(max_compounds=50)
    
    if not mdm2_data.empty:
        # Save to file
        collector.save_dataset(mdm2_data, "../../data/raw/mdm2_test_data.csv")
        
        print(f"\nDataset Summary:")
        print(f"Total compounds: {len(mdm2_data)}")
        print(f"Inhibitors: {mdm2_data['is_inhibitor'].sum()}")
        print(f"Non-inhibitors: {len(mdm2_data) - mdm2_data['is_inhibitor'].sum()}")
        
        print(f"\nColumns available:")
        print(mdm2_data.columns.tolist())
        
        print(f"\nFirst few rows:")
        print(mdm2_data[['molecule_chembl_id', 'canonical_smiles', 'pActivity', 'is_inhibitor']].head())
    else:
        print("No data collected")