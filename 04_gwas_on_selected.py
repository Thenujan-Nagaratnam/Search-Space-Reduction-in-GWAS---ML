import subprocess
import os
import pandas as pd

# --- Configuration ---
# Data directories
qc_data_dir = 'qc_data' # Contains per-population QC'd files (e.g., _s2_mind for pre-LD prune, _s3_ldpruned_final for post-LD prune)
ml_results_dir = 'ml_results' # Contains selected SNP lists
pheno_files_dir = 'prepared_data' # Contains phenotype and covariate files
gwas_output_dir = 'gwas_results'
os.makedirs(gwas_output_dir, exist_ok=True)

# Population list
populations_to_process = ['Chinese', 'Malay', 'Indian']

# Phenotype and Covariate files
# This must match the header name for the phenotype in your .txt files (e.g. 'TARGET_LIPID')
PHENOTYPE_NAME_IN_FILE = 'Cholesterol' 
# Covariate file (contains FID, IID, and PCs from script 02)
COVARIATE_FILE = os.path.join(pheno_files_dir, 'covariates.txt')
# Specify which covariates to use from the file if it contains more than just PCs (e.g., 'PC1 PC2 PC3 Age Sex')
# For now, assume we use all PC columns found. Script will determine PC columns.
# If you have other covariates like 'Age', 'Sex' in covariates.txt, list them here:
# COVARIATE_NAMES = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'Age', 'Sex'] # Example
COVARIATE_NAMES = None # Will be auto-detected for PCs

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
            # Read the entire error output
            error_output = process.stderr.read() if process.stderr else ""
            print(f"ERROR: PLINK command failed with exit code {process.returncode}. See {log_filename}")
            print(f"PLINK error output: {error_output}")  # Print the error
            return False
        print(f"PLINK command successful. Log: {log_filename}")
        return True
    except FileNotFoundError:
        print(f"ERROR: PLINK executable ('{PLINK_EXECUTABLE}') not found.")
        return False
    except Exception as e:
        print(f"ERROR running PLINK: {e}")
        return False

# --- Determine Covariate Names if not specified (for PCs) ---
if COVARIATE_NAMES is None and os.path.exists(COVARIATE_FILE):
    try:
        cov_df_header = pd.read_csv(COVARIATE_FILE, sep='\t', nrows=0) # Read only header
        COVARIATE_NAMES = [col for col in cov_df_header.columns if col.startswith('PC')]
        if COVARIATE_NAMES:
            print(f"Auto-detected PC covariates to use: {', '.join(COVARIATE_NAMES)}")
        else:
            print("No PC covariates auto-detected in covariate file. GWAS on merged data will not use PCs unless specified.")
    except Exception as e:
        print(f"Could not read covariate file header to auto-detect PCs: {e}")
        COVARIATE_NAMES = [] # Ensure it's a list
elif COVARIATE_NAMES is None:
    COVARIATE_NAMES = [] # Ensure it's a list if file doesn't exist

# --- Option 1: GWAS on ML-selected SNPs for Merged Population (with PCs) ---
print("\n--- GWAS on ML-selected SNPs (Merged Population with PCs) ---")
selected_snps_union_file = os.path.join(ml_results_dir, "all_populations_selected_snps_rf_union.txt")
merged_pheno_file = os.path.join(pheno_files_dir, 'merged_pheno.txt')

# For merged GWAS, we need a merged PLINK fileset.
# This should be based on data *before* per-population LD pruning, but *after* basic QC (MAF, HWE, geno, mind).
# The 'merged_for_pca_unpruned' from script 02 (which was s2_mind files merged) is suitable.
merged_bfile_for_gwas = os.path.join(qc_data_dir, '..', 'pca_data', 'merged_for_pca_unpruned') # Path from script 02 output

if os.path.exists(selected_snps_union_file) and \
   os.path.exists(f"{merged_bfile_for_gwas}.bed") and \
   os.path.exists(merged_pheno_file) and \
   os.path.exists(COVARIATE_FILE) and COVARIATE_NAMES:

    gwas_ml_merged_out = os.path.join(gwas_output_dir, 'gwas_ml_selected_merged')
    cmd_gwas_ml_merged = [PLINK_EXECUTABLE, '--bfile', merged_bfile_for_gwas,
                          '--extract', selected_snps_union_file,
                          '--pheno', merged_pheno_file, '--pheno-name', PHENOTYPE_NAME_IN_FILE,
                          '--covar', COVARIATE_FILE, '--covar-name', *COVARIATE_NAMES,
                          '--assoc', 'qt-means', # Start with the most basic --assoc
                          '--out', gwas_ml_merged_out]
    run_plink_command(cmd_gwas_ml_merged, f"{gwas_output_dir}/log_gwas_ml_merged")
else:
    print("Skipping GWAS on ML-selected SNPs (Merged Population): Missing input files or covariates.")
    if not os.path.exists(selected_snps_union_file): print(f" - Missing: {selected_snps_union_file}")
    if not os.path.exists(f"{merged_bfile_for_gwas}.bed"): print(f" - Missing: {merged_bfile_for_gwas}.bed")
    if not os.path.exists(merged_pheno_file): print(f" - Missing: {merged_pheno_file}")
    if not os.path.exists(COVARIATE_FILE): print(f" - Missing: {COVARIATE_FILE}")
    if not COVARIATE_NAMES: print(f" - Missing: Auto-detected PC covariates.")


# --- Option 2: Traditional GWAS on Merged Population (LD-pruned, with PCs) ---
# This uses the LD-pruned merged dataset prepared for PCA in script 02.
print("\n--- Traditional GWAS (LD-pruned Merged Population with PCs) ---")
merged_ldpruned_bfile_for_gwas = os.path.join(qc_data_dir, '..', 'pca_data', 'merged_for_pca_ldpruned_final') # From script 02

if os.path.exists(f"{merged_ldpruned_bfile_for_gwas}.bed") and \
   os.path.exists(merged_pheno_file) and \
   os.path.exists(COVARIATE_FILE) and COVARIATE_NAMES:

    gwas_trad_merged_out = os.path.join(gwas_output_dir, 'gwas_traditional_merged_ldpruned')
    cmd_gwas_trad_merged = [PLINK_EXECUTABLE, '--bfile', merged_ldpruned_bfile_for_gwas,
                            '--pheno', merged_pheno_file, '--pheno-name', PHENOTYPE_NAME_IN_FILE,
                            '--covar', COVARIATE_FILE, '--covar-name', *COVARIATE_NAMES,
                            '--assoc', 'qt-means', # Start with the most basic --assoc
                            '--out', gwas_trad_merged_out]
    run_plink_command(cmd_gwas_trad_merged, f"{gwas_output_dir}/log_gwas_trad_merged")
else:
    print("Skipping Traditional GWAS (Merged Population): Missing input files or covariates.")
    if not os.path.exists(f"{merged_ldpruned_bfile_for_gwas}.bed"): print(f" - Missing: {merged_ldpruned_bfile_for_gwas}.bed")
    # Other checks are same as above.


# --- Option 3: Per-Population GWAS (on ML-selected SNPs for each pop) ---
# This demonstrates running GWAS separately if desired.
# Covariates like Age/Sex could be used if prepared, but PCs are typically for merged data.
# For simplicity, this example runs without additional per-population covariates, but they could be added.
print("\n--- GWAS on ML-selected SNPs (Per-Population) ---")
for pop in populations_to_process:
    print(f"\nProcessing {pop} for per-population GWAS...")
    selected_snps_pop_file = os.path.join(ml_results_dir, f"{pop.lower()}_selected_snps_rf.txt")
    # Use the per-population data *after basic QC but before per-pop LD pruning*
    # This is because ML was done on LD-pruned, but for GWAS itself on *selected* SNPs, LD isn't the primary concern
    # The _s2_mind files are suitable here.
    pop_bfile_for_gwas = os.path.join(qc_data_dir, f"{pop.lower()}_s2_mind")
    pop_pheno_file = os.path.join(pheno_files_dir, f"{pop.lower()}_pheno.txt")

    if os.path.exists(selected_snps_pop_file) and \
       os.path.exists(f"{pop_bfile_for_gwas}.bed") and \
       os.path.exists(pop_pheno_file):
        
        gwas_ml_pop_out = os.path.join(gwas_output_dir, f'gwas_ml_selected_{pop.lower()}')
        cmd_gwas_ml_pop = [PLINK_EXECUTABLE, '--bfile', pop_bfile_for_gwas,
                            '--extract', selected_snps_pop_file,
                            '--pheno', pop_pheno_file, '--pheno-name', PHENOTYPE_NAME_IN_FILE,
                            '--assoc', 'qt-means', # Start with the most basic --assoc
                            '--out', gwas_ml_pop_out]
        # If you have per-population covariates (e.g. Age, Sex, but NOT cross-population PCs)
        # you would add them here using a per-population covariate file.
        run_plink_command(cmd_gwas_ml_pop, f"{gwas_output_dir}/log_gwas_ml_{pop.lower()}")
    else:
        print(f"Skipping GWAS on ML-selected SNPs for {pop}: Missing input files.")
        if not os.path.exists(selected_snps_pop_file): print(f" - Missing: {selected_snps_pop_file}")
        if not os.path.exists(f"{pop_bfile_for_gwas}.bed"): print(f" - Missing: {pop_bfile_for_gwas}.bed")
        if not os.path.exists(pop_pheno_file): print(f" - Missing: {pop_pheno_file}")

print("\n--- GWAS Script Finished ---")
print(f"GWAS results are in: {gwas_output_dir}")
