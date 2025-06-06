# Search Space Reduction in GWAS using Machine Learning

please find the dataset, implementation and the results here: https://github.com/Thenujan-Nagaratnam/Search-Space-Reduction-in-GWAS---ML

This project implements a sophisticated bioinformatics pipeline that combines machine learning techniques with Genome-Wide Association Studies (GWAS) to efficiently reduce the search space for genetic variants. The pipeline processes multi-ethnic lipidomic data and integrates various quality control measures, machine learning-based feature selection, and comprehensive visualization tools.

## Overview

The pipeline consists of six main steps:

1. **Data Preparation** (`01_data_preparation.py`)

   - Processes lipidomic data from multiple ethnic groups (Chinese, Malay, Indian)
   - Handles sample ID mapping
   - Standardizes phenotype measurements
   - Creates PLINK-compatible phenotype files

2. **Quality Control** (`02_quality_control.py`)

   - Performs genetic data quality control
   - Implements standard GWAS QC steps
   - Handles population stratification

3. **ML Feature Selection** (`03_ml_feature_selection.py`)

   - Implements machine learning algorithms for feature selection
   - Reduces the genetic variant search space
   - Uses SHAP values for feature importance

4. **GWAS Analysis** (`04_gwas_on_selected.py`)

   - Performs GWAS on the selected subset of variants
   - Uses PLINK for association testing
   - Processes results for interpretation

5. **Visualization** (`05_visualization.py`)

   - Creates various plots and visualizations
   - Generates SHAP plots for ML interpretability
   - Produces publication-ready figures

6. **Validation** (`06_validation.py`)
   - Performs comprehensive model validation:
     - Cross-validation (5-fold) for robust performance estimation
     - Train-test split evaluation (80-20 split)
     - Comparative analysis between ML-selected SNPs and all LD-pruned SNPs
   - Generates performance metrics:
     - R² scores for training and test sets
     - Mean Squared Error (MSE) for training and test sets
     - Cross-validation scores with standard deviations
   - Creates visualization plots:
     - Performance comparison plots between models
     - Test R² and MSE comparisons across populations
   - Saves validated models for future use

## Project Structure

```
.
├── 01_data_preparation.py
├── 02_quality_control.py
├── 03_ml_feature_selection.py
├── 04_gwas_on_selected.py
├── 05_visualization.py
├── 06_validation.py
├── public/                  # Raw data directory
├── plink/                  # Genome association analysis toolset
├── prepared_data/          # Processed phenotype and covariate files
├── qc_data/               # Quality-controlled genetic data
├── pca_data/             # Principal Component Analysis results
├── ml_results/           # Machine learning outputs
├── gwas_results/         # GWAS analysis results
├── visualization_plots/  # Generated visualizations
├── validation_results/   # Validation outputs and metrics
└── requirements.txt
```

## Requirements

### Software Dependencies

- Python 3.x
- PLINK 1.9 - Download from https://www.cog-genomics.org/plink/ and put it in /plink folder

### Python Packages

- pandas
- numpy
- scikit-learn
- joblib
- shap
- matplotlib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Search-Space-Reduction-in-GWAS---ML-1.git
cd Search-Space-Reduction-in-GWAS---ML-1
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Ensure PLINK 1.9 is installed and accessible in your system PATH.

## Data Requirements

The pipeline expects the following data structure in the `public` directory:

```
public/
├── Lipidomic/
│   ├── 122Chinese_282lipids.txt
│   ├── 117Malay_282lipids.txt
│   └── 120Indian_282lipids.txt
├── Genomics/
│   ├── 110Chinese_2527458snps.fam
│   ├── 108Malay_2527458snps.fam
│   └── 105Indian_2527458snps.fam
├── Phenotype/
│   └── IOmics_Questionnaire_HealthScreening_and_FFQ.csv
└── Info/
    └── iomics_ID.csv
```

## Usage

Run the pipeline scripts in sequence:

1. Prepare the data:

```bash
python 01_data_preparation.py
```

2. Perform quality control:

```bash
python 02_quality_control.py
```

3. Run machine learning feature selection:

```bash
python 03_ml_feature_selection.py
```

4. Conduct GWAS on selected features:

```bash
python 04_gwas_on_selected.py
```

5. Generate visualizations:

```bash
python 05_visualization.py
```

6. Validate results:

```bash
python 06_validation.py
```

The validation script will:

- Compare ML-selected SNPs against all LD-pruned SNPs
- Generate performance metrics for each population
- Create visualization plots in the validation_results directory
- Save trained models for future use

## Features

- Automated pipeline with logging and error handling
- Integration with PLINK for genetic data processing
- Multi-ethnic analysis support
- Machine learning-based feature selection
- Comprehensive visualization tools
- SHAP analysis for ML interpretability
- Robust validation framework:
  - Cross-validation assessment
  - Population-specific performance metrics
  - Comparative analysis of feature selection effectiveness
  - Model persistence for reproducibility

## Dataset Reference

The analysis is based on the dataset from:
https://academic.oup.com/hmg/article/30/7/603/6129656

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/Search-Space-Reduction-in-GWAS---ML-1
