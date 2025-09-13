# P53-MDM2 Drug Design Prediction Model

## Overview

This project implements a foundational machine learning model designed to predict p53-MDM2 protein-protein interaction behavior with unknown drug compounds. The model leverages a comprehensive dataset of 800 known ChEMBL drugs to establish baseline predictions and is enhanced through molecular docking simulations to expand the training dataset.

## Approach

### Data Foundation
- **Known Drug Dataset**: 800 ChEMBL drugs with established p53-MDM2 interaction profiles
- **Molecular Docking**: Small molecule docking simulations generate additional training data through docking poses
- **Feature Selection**: Evolutionary algorithms optimize molecular descriptors using Pareto curve analysis to identify the top 30 most predictive features

### Model Architecture
- **Graph Neural Networks (GNNs)**: Capture molecular structure and interaction patterns
- **Adversarial Networks**: Enhance model robustness and generalization capabilities
- **Hybrid Architecture**: Combines both approaches for improved prediction accuracy

## Methodology

1. **Data Collection**: Curate 800 ChEMBL drugs with known p53-MDM2 activity
2. **Molecular Docking**: Perform systematic docking of small molecules to generate diverse binding poses
3. **Feature Engineering**: Extract molecular descriptors from both known drugs and docked compounds
4. **Feature Optimization**: Apply evolutionary algorithms to identify optimal descriptor subset (≤30 features)
5. **Model Training**: Train hybrid GNN-adversarial network architecture
6. **Validation**: Evaluate model performance on unknown drug compounds

## Key Features

- Prediction of p53-MDM2 interaction behavior for novel compounds
- Evolutionary algorithm-based feature selection for optimal descriptor identification
- Integration of molecular docking data to enhance training diversity
- Hybrid neural network architecture combining graph-based and adversarial learning
- Pareto-optimal feature selection ensuring model efficiency and interpretability

## Applications

This model enables rapid screening and prediction of drug candidates targeting the p53-MDM2 interaction, accelerating the drug discovery process for cancer therapeutics targeting this critical pathway.