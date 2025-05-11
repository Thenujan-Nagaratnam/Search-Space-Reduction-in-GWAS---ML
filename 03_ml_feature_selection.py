# 03_ml_feature_selection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # Using Regressor for quantitative lipid trait
# from sklearn.model_selection import train_test_split # Not explicitly used for splitting here, full data for RF importance
import subprocess
import os
import joblib # For saving the model
import shap # For SHAP interpretation
import matplotlib.pyplot as plt # For SHAP plots

# --- Configuration ---
qc_data_dir = 'qc_data' # From 02_quality_control.py
pheno_files_dir = 'prepared_data' # From 01_data_preparation.py

# Per-population QC'd and LD-pruned filesets (prefixes)
# These are the outputs like 'qc_data/chinese_s3_ldpruned_final'
# This script will look for these based on convention.
populations_to_process = ['Chinese', 'Malay', 'Indian']

# Phenotype column name in the phenotype files
# Must match the header in files like 'chinese_pheno.txt' (e.g., TARGET_PHENOTYPE_COL from script 01)
PHENOTYPE_NAME_IN_FILE = 'Cholesterol' # This was set as header in 01_data_preparation.py

# ML parameters
N_ESTIMATORS_RF = 100
RANDOM_STATE_RF = 42
NUM_SNPS_TO_SELECT_PER_POP = 500 # Example: select top 500 SNPs per population
# SHAP parameters
NUM_SNPS_FOR_SHAP_SUMMARY = 20 # Number of top SNPs to show in SHAP summary plot

# Output directory for ML results
ml_output_dir = 'ml_results'
os.makedirs(ml_output_dir, exist_ok=True)
shap_plots_dir = os.path.join(ml_output_dir, 'shap_plots') # Subdirectory for SHAP plots
os.makedirs(shap_plots_dir, exist_ok=True)


PLINK_EXECUTABLE = 'plink/plink.exe' # Or full path

def run_plink_command(command_args, log_prefix):
    """Runs a PLINK command and logs output."""
    try:
        print(f"Running PLINK: {' '.join(command_args)}")
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_filename = f"{log_prefix}.log"
        with open(log_filename, 'w') as log_file:
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"ERROR: PLINK command failed with exit code {process.returncode}. See {log_filename}")
            return False
        print(f"PLINK command successful. Log: {log_filename}")
        return True
    except FileNotFoundError:
        print(f"ERROR: PLINK executable ('{PLINK_EXECUTABLE}') not found.")
        return False
    except Exception as e:
        print(f"ERROR running PLINK: {e}")
        return False

# --- ML Feature Selection for each population ---
all_selected_snps_combined = set()

for pop in populations_to_process:
    print(f"\n--- Starting ML Feature Selection for {pop} ---")
    
    # Define input files for this population
    # This is the QC'd, LD-pruned fileset from script 02
    bfile_prefix_ml = os.path.join(qc_data_dir, f"{pop.lower()}_s3_ldpruned_final")
    pheno_file_path = os.path.join(pheno_files_dir, f"{pop.lower()}_pheno.txt")
    
    if not (os.path.exists(f"{bfile_prefix_ml}.bed") and os.path.exists(pheno_file_path)):
        print(f"Data or phenotype file for {pop} not found. Skipping ML for this population.")
        print(f"Expected bfile: {bfile_prefix_ml}.bed, Expected pheno: {pheno_file_path}")
        continue
        
    # 1. Convert PLINK to numerical format for ML (.raw)
    raw_file_prefix = os.path.join(ml_output_dir, f"{pop.lower()}_for_ml")
    cmd_recode_a = [PLINK_EXECUTABLE, '--bfile', bfile_prefix_ml,
                    '--pheno', pheno_file_path, '--pheno-name', PHENOTYPE_NAME_IN_FILE, # Ensure phenotype is loaded
                    '--recode', 'A', # 'A' for additive model (0,1,2 dosages), 'header' for SNP IDs in .raw
                    '--out', raw_file_prefix]
    if not run_plink_command(cmd_recode_a, f"{ml_output_dir}/{pop.lower()}_log_recode_a"):
        print(f"Failed to generate .raw file for {pop}. Skipping ML.")
        continue
    
    raw_file_path = f"{raw_file_prefix}.raw"
    if not os.path.exists(raw_file_path):
        print(f".raw file not found at {raw_file_path} for {pop}. Skipping ML.")
        continue
        
    # 2. Load data into pandas
    try:
        geno_df = pd.read_csv(raw_file_path, sep='\s+')
        # PHENOTYPE column in .raw is the one PLINK used from --pheno
        # FID, IID, PAT, MAT, SEX, PHENOTYPE, SNP1_A, SNP2_C ...
        y = geno_df['PHENOTYPE'] # This is already the scaled phenotype
        X = geno_df.drop(columns=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], errors='ignore')
        # Clean SNP names (remove _ALLELE if present, e.g. SNP1_A -> SNP1)
        X.columns = [col.split('_')[0] for col in X.columns]

    except Exception as e:
        print(f"Error loading or processing .raw file for {pop}: {e}")
        continue
        
    if X.empty or y.isnull().all():
        print(f"No genotype data or all phenotype data is missing for {pop} after loading. Skipping.")
        continue

    # Handle cases where phenotype might still have NaNs if PLINK couldn't use it (e.g. all missing)
    if y.isnull().any():
        print(f"Warning: NaNs found in phenotype for {pop} from .raw file. Dropping these samples for ML.")
        valid_indices = ~y.isnull()
        X = X[valid_indices].reset_index(drop=True) # Reset index after filtering
        y = y[valid_indices].reset_index(drop=True) # Reset index after filtering
        if X.empty:
            print(f"No samples left for {pop} after dropping NaN phenotypes. Skipping.")
            continue
            
    # 3. Train Random Forest Regressor
    print(f"Training RandomForestRegressor for {pop} on {X.shape[0]} samples and {X.shape[1]} SNPs...")
    rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE_RF, n_jobs=-1)
    rf_model.fit(X, y)
    
    # Save the trained model
    model_filename = os.path.join(ml_output_dir, f"{pop.lower()}_rf_model.joblib")
    joblib.dump(rf_model, model_filename)
    print(f"Saved trained RF model for {pop} to {model_filename}")

    # 4. Get feature importances from Random Forest
    importances_rf = rf_model.feature_importances_
    feature_importance_df_rf = pd.DataFrame({'SNP': X.columns, 'Importance_RF': importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance_RF', ascending=False)
    
    print(f"\nTop 10 most important SNPs for {pop} (from RF):")
    print(feature_importance_df_rf.head(10))
    
    # 5. Select top N SNPs based on RF importance
    selected_snps_pop_rf = feature_importance_df_rf.head(NUM_SNPS_TO_SELECT_PER_POP)['SNP'].tolist()
    print(f"Selected {len(selected_snps_pop_rf)} SNPs for {pop} using Random Forest for downstream GWAS.")
    
    # Save the list of RF-selected SNPs for this population
    selected_snps_filepath_rf = os.path.join(ml_output_dir, f"{pop.lower()}_selected_snps_rf.txt")
    with open(selected_snps_filepath_rf, 'w') as f:
        for snp_id in selected_snps_pop_rf:
            f.write(f"{snp_id}\n")
    print(f"Saved list of RF-selected SNPs for {pop} to {selected_snps_filepath_rf}")
    
    all_selected_snps_combined.update(selected_snps_pop_rf)

    # --- SHAP Interpretation ---
    print(f"\nCalculating SHAP values for {pop}...")
    try:
        # For TreeExplainer, it's good practice to pass the model and the data it was trained on,
        # or a representative background dataset if X is very large.
        # Here, X is the training data.
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values. Can be slow for large datasets / many features.
        # Consider using a subset of X for shap_values if performance is an issue:
        # X_subset_for_shap = shap.sample(X, 100) # e.g., 100 samples
        # shap_values = explainer.shap_values(X_subset_for_shap)
        # For now, using full X, assuming it's manageable for the iOmics cohort sizes.
        shap_values = explainer.shap_values(X) # For regressors, this is a single array

        # Create a SHAP summary plot (beeswarm or bar)
        # Beeswarm plot shows feature importance and effect
        plt.figure() # Create a new figure to avoid overlap if run in a loop interactively
        shap.summary_plot(shap_values, X, plot_type="beeswarm", max_display=NUM_SNPS_FOR_SHAP_SUMMARY, show=False)
        plt.title(f"SHAP Summary Plot (Beeswarm) - {pop}", fontsize=10)
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(shap_plots_dir, f"{pop.lower()}_shap_beeswarm.png")
        plt.savefig(shap_beeswarm_path)
        plt.close() # Close the plot figure
        print(f"Saved SHAP beeswarm plot to {shap_beeswarm_path}")

        # Bar plot for global feature importance from SHAP
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=NUM_SNPS_FOR_SHAP_SUMMARY, show=False)
        plt.title(f"SHAP Summary Plot (Bar) - {pop}", fontsize=10)
        plt.tight_layout()
        shap_bar_path = os.path.join(shap_plots_dir, f"{pop.lower()}_shap_bar.png")
        plt.savefig(shap_bar_path)
        plt.close()
        print(f"Saved SHAP bar plot to {shap_bar_path}")
        
        # You can also create a DataFrame of mean absolute SHAP values for ranking
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df_shap = pd.DataFrame({'SNP': X.columns, 'Importance_SHAP': mean_abs_shap})
        feature_importance_df_shap = feature_importance_df_shap.sort_values(by='Importance_SHAP', ascending=False)
        
        shap_importance_filepath = os.path.join(ml_output_dir, f"{pop.lower()}_feature_importance_shap.csv")
        feature_importance_df_shap.to_csv(shap_importance_filepath, index=False)
        print(f"Saved SHAP feature importances to {shap_importance_filepath}")
        print(f"\nTop 10 most important SNPs for {pop} (from SHAP):")
        print(feature_importance_df_shap.head(10))

    except Exception as e:
        print(f"ERROR during SHAP analysis for {pop}: {e}")
        print("Ensure 'shap' and 'matplotlib' are installed. You might also need a C++ compiler for some SHAP backends.")


# Save the combined list of all unique SNPs selected across all populations (from RF)
if all_selected_snps_combined:
    combined_selected_snps_filepath = os.path.join(ml_output_dir, "all_populations_selected_snps_rf_union.txt")
    with open(combined_selected_snps_filepath, 'w') as f:
        for snp_id in sorted(list(all_selected_snps_combined)): # Sort for consistency
            f.write(f"{snp_id}\n")
    print(f"\nSaved combined (union) list of {len(all_selected_snps_combined)} RF-selected SNPs to {combined_selected_snps_filepath}")
else:
    print("\nNo SNPs were selected by ML from any population.")


print("\n--- ML Feature Selection Script Finished ---")