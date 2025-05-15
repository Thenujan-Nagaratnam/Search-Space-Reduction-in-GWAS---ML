# 06_validation.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Directories
ml_output_dir = 'ml_results'
qc_data_dir = 'qc_data'
validation_output_dir = 'validation_results'
os.makedirs(validation_output_dir, exist_ok=True)

# Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS_RF = 100  # Same as in script 03
N_CV_FOLDS = 5

# Population info
populations_to_process = ['Chinese', 'Malay', 'Indian']
PHENOTYPE_NAME_IN_FILE = 'Cholesterol'

# PLINK executable path
PLINK_EXECUTABLE = 'plink/plink.exe'

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
    except Exception as e:
        print(f"ERROR running PLINK: {e}")
        return False

def load_and_prepare_data(bfile_prefix, snps_to_extract=None):
    """Load genetic data from PLINK files and prepare for ML."""
    # Create temporary directory for intermediate files
    os.makedirs('temp', exist_ok=True)
    temp_prefix = os.path.join('temp', 'temp_data')
    
    # Get the population from the bfile path
    pop = os.path.basename(bfile_prefix).split('_')[0]
    pheno_file = os.path.join('prepared_data', f'{pop}_pheno.txt')
    
    # Base command to recode data
    cmd = [PLINK_EXECUTABLE, '--bfile', bfile_prefix,
           '--pheno', pheno_file,  # Add phenotype file
           '--pheno-name', PHENOTYPE_NAME_IN_FILE]  # Specify phenotype column
    
    # If specific SNPs are provided, extract only those
    if snps_to_extract is not None:
        temp_snp_file = os.path.join('temp', 'snps_to_extract.txt')
        with open(temp_snp_file, 'w') as f:
            for snp in snps_to_extract:
                f.write(f"{snp}\n")
        cmd.extend(['--extract', temp_snp_file])
    
    # Add recode command
    cmd.extend(['--recode', 'A', '--out', temp_prefix])
    
    if not run_plink_command(cmd, os.path.join('temp', 'recode_log')):
        raise Exception("Failed to recode PLINK data")
    
    # Load the recoded data
    raw_file = f"{temp_prefix}.raw"
    data = pd.read_csv(raw_file, sep='\s+')
    
    # Extract features and target
    X = data.drop(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], axis=1, errors='ignore')
    y = data['PHENOTYPE']  # This should now contain the correct phenotype values
    
    # Clean SNP names
    X.columns = [col.split('_')[0] for col in X.columns]
    
    return X, y

def evaluate_model(X, y, model_name):
    """Evaluate model using cross-validation and train-test split."""
    # Train-test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS_RF, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    
    # Train model
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    # Cross-validation scores
    cv_r2_scores = cross_val_score(rf, X, y, cv=N_CV_FOLDS, scoring='r2')
    cv_mse_scores = -cross_val_score(rf, X, y, cv=N_CV_FOLDS, scoring='neg_mean_squared_error')
    
    results = {
        'Model': model_name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'CV R² (mean ± std)': f"{cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}",
        'CV MSE (mean ± std)': f"{cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}",
        'N Features': X.shape[1]
    }
    
    return results, rf

# Results storage
all_results = []

# Process each population
for pop in populations_to_process:
    print(f"\n--- Evaluating models for {pop} population ---")
    
    # Load ML-selected SNPs
    selected_snps_file = os.path.join(ml_output_dir, f"{pop.lower()}_selected_snps_rf.txt")
    if not os.path.exists(selected_snps_file):
        print(f"Selected SNPs file not found for {pop}. Skipping.")
        continue
    
    with open(selected_snps_file, 'r') as f:
        selected_snps = [line.strip() for line in f]
    
    # Path to LD-pruned dataset
    ld_pruned_bfile = os.path.join(qc_data_dir, f"{pop.lower()}_s3_ldpruned_final")
    
    try:
        # Evaluate model with ML-selected SNPs
        print(f"Evaluating model with {len(selected_snps)} ML-selected SNPs...")
        X_selected, y = load_and_prepare_data(ld_pruned_bfile, selected_snps)
        results_selected, model_selected = evaluate_model(X_selected, y, f"{pop} - ML Selected SNPs")
        all_results.append(results_selected)
        
        # Evaluate model with all LD-pruned SNPs
        print("Evaluating model with all LD-pruned SNPs...")
        X_all, y = load_and_prepare_data(ld_pruned_bfile)
        results_all, model_all = evaluate_model(X_all, y, f"{pop} - All LD-pruned SNPs")
        all_results.append(results_all)
        
        # Save models
        joblib.dump(model_selected, os.path.join(validation_output_dir, f"{pop.lower()}_model_selected_snps.joblib"))
        joblib.dump(model_all, os.path.join(validation_output_dir, f"{pop.lower()}_model_all_ldpruned.joblib"))
        
    except Exception as e:
        print(f"Error processing {pop}: {e}")
        continue

# Create results DataFrame and save
results_df = pd.DataFrame(all_results)
results_file = os.path.join(validation_output_dir, 'model_comparison_results.csv')
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

# Create comparison plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, x='Model', y='Test R²')
plt.xticks(rotation=45, ha='right')
plt.title('Test R² Comparison')

plt.subplot(1, 2, 2)
sns.barplot(data=results_df, x='Model', y='Test MSE')
plt.xticks(rotation=45, ha='right')
plt.title('Test MSE Comparison')

plt.tight_layout()
plt.savefig(os.path.join(validation_output_dir, 'performance_comparison.png'))
plt.close()

# Print summary
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Clean up temporary files
import shutil
if os.path.exists('temp'):
    shutil.rmtree('temp') 
