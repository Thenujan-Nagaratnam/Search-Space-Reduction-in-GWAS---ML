# 01_data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re  # For cleaning lipid names

# --- Configuration ---
BASE_DATA_PATH = "public"  # Assuming 'public' folder is in the same dir as script

lipid_files_info = {
    "Chinese": {
        "file": os.path.join(BASE_DATA_PATH, "Lipidomic", "122Chinese_282lipids.txt"),
        "fam": os.path.join(BASE_DATA_PATH, "Genomics", "110Chinese_2527458snps.fam"),
    },
    "Malay": {
        "file": os.path.join(BASE_DATA_PATH, "Lipidomic", "117Malay_282lipids.txt"),
        "fam": os.path.join(BASE_DATA_PATH, "Genomics", "108Malay_2527458snps.fam"),
    },
    "Indian": {
        "file": os.path.join(BASE_DATA_PATH, "Lipidomic", "120Indian_282lipids.txt"),
        "fam": os.path.join(BASE_DATA_PATH, "Genomics", "105Indian_2527458snps.fam"),
    },
}

# Path to the general phenotype questionnaire (for covariates like Age, Sex if needed later)
questionnaire_file = os.path.join(
    BASE_DATA_PATH, "Phenotype", "IOmics_Questionnaire_HealthScreening_and_FFQ.csv"
)
sample_id_mapping_file = os.path.join(BASE_DATA_PATH, "Info", "iomics_ID.csv")

output_dir = "prepared_data"
os.makedirs(output_dir, exist_ok=True)

# --- Target Phenotype (from lipidomic data) ---
# USER ACTION: Please verify this lipid name or choose a different one from your lipid data files.
# The R script used "Cholesterol" generically. Lipid files usually have specific species.
# Example: 'TG(52:2)' or if a summary measure like 'Total Cholesterol' exists.
# If set to None, the script will try to list available lipids and pick the first one.
TARGET_LIPID_NAME_FROM_FILE = "Cholesterol"  # e.g., 'Cholesterol' or 'TG(52:2)'
# This will be the cleaned name for use as a column header:
TARGET_PHENOTYPE_COL = "Cholesterol"  # Default if specific name isn't processed

# --- 1. Load and Merge Lipidomic Data ---
all_lipid_data_frames = []
print("Loading lipidomic data...")
available_lipid_columns = set()

for ethnic_group, info in lipid_files_info.items():
    file_path = info["file"]
    try:
        df = pd.read_csv(
            file_path, sep="\t", index_col=0
        )  # Samples as rows, Lipids as columns
        df.index.name = "SampleID_Lipid"
        df["ethnic_group"] = ethnic_group
        all_lipid_data_frames.append(df)
        if not available_lipid_columns:  # Get columns from the first file
            available_lipid_columns.update(
                col for col in df.columns if col != "ethnic_group"
            )
        else:  # Intersect to find common columns, though R script implies they are the same
            available_lipid_columns.intersection_update(
                col for col in df.columns if col != "ethnic_group"
            )
        print(
            f"Loaded {ethnic_group} lipid data: {df.shape[0]} samples, {len(df.columns)-1} lipids."
        )
    except FileNotFoundError:
        print(f"ERROR: Lipid file not found: {file_path}")
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")

if not all_lipid_data_frames:
    print("CRITICAL ERROR: No lipid data loaded. Exiting.")
    exit()

merged_lipid_df = pd.concat(all_lipid_data_frames)
print(f"\nMerged lipid data: {merged_lipid_df.shape[0]} samples initially.")

if not available_lipid_columns:
    print(
        "CRITICAL ERROR: No common lipid columns found across files or no files loaded properly."
    )
    exit()

# --- Determine and Prepare Target Phenotype ---
if TARGET_LIPID_NAME_FROM_FILE is None:
    print("\nTARGET_LIPID_NAME_FROM_FILE not specified.")
    print("Available common lipid columns (example from first file type):")
    for i, lipid_col in enumerate(list(available_lipid_columns)[:10]):  # Show first 10
        print(f"  {i+1}. {lipid_col}")
    if available_lipid_columns:
        TARGET_LIPID_NAME_FROM_FILE = list(available_lipid_columns)[
            0
        ]  # Pick the first one
        print(f"Using first available lipid as target: '{TARGET_LIPID_NAME_FROM_FILE}'")
    else:
        print("CRITICAL ERROR: No lipid columns available to choose as target.")
        exit()

if TARGET_LIPID_NAME_FROM_FILE not in merged_lipid_df.columns:
    print(
        f"CRITICAL ERROR: Specified/chosen TARGET_LIPID_NAME_FROM_FILE ('{TARGET_LIPID_NAME_FROM_FILE}') not in merged data."
    )
    exit()

# Clean the lipid name to be a valid column name (e.g., remove special chars)
TARGET_PHENOTYPE_COL = re.sub(r"[^A-Za-z0-9_]+", "_", TARGET_LIPID_NAME_FROM_FILE)
merged_lipid_df[TARGET_PHENOTYPE_COL] = pd.to_numeric(
    merged_lipid_df[TARGET_LIPID_NAME_FROM_FILE], errors="coerce"
)

nan_count_before_impute = merged_lipid_df[TARGET_PHENOTYPE_COL].isnull().sum()
if nan_count_before_impute > 0:
    mean_pheno = merged_lipid_df[TARGET_PHENOTYPE_COL].mean()
    merged_lipid_df[TARGET_PHENOTYPE_COL].fillna(mean_pheno, inplace=True)
    print(
        f"Imputed {nan_count_before_impute} NaNs in '{TARGET_PHENOTYPE_COL}' with mean ({mean_pheno:.4f})."
    )

scaler = StandardScaler()
scaled_phenotype_col_name = f"{TARGET_PHENOTYPE_COL}_scaled"
merged_lipid_df[scaled_phenotype_col_name] = scaler.fit_transform(
    merged_lipid_df[[TARGET_PHENOTYPE_COL]]
)
print(
    f"Target phenotype '{TARGET_PHENOTYPE_COL}' (from '{TARGET_LIPID_NAME_FROM_FILE}') has been scaled into '{scaled_phenotype_col_name}'."
)


# --- 2. Load Sample ID Mapping File ---
# USER ACTION: Verify these column names in your iomics_ID.csv file!
# These names should correspond to:
# ID_COL_FOR_LIPID_DATA: The column in iomics_ID.csv that contains IDs matching the row names of your lipid files.
# ID_COL_FOR_GENOMIC_DATA: The column in iomics_ID.csv that contains IDs matching the IID (2nd col) in your .fam files.
ID_COL_FOR_LIPID_DATA = "LIPIDOMIC_ID"  # Placeholder - VERIFY THIS!
ID_COL_FOR_GENOMIC_DATA = "GWAS_ID"  # Placeholder - VERIFY THIS!

try:
    id_map_df = pd.read_csv(sample_id_mapping_file)
    print(f"\nLoaded sample ID mapping file: {sample_id_mapping_file}")
    if not (
        ID_COL_FOR_LIPID_DATA in id_map_df.columns
        and ID_COL_FOR_GENOMIC_DATA in id_map_df.columns
    ):
        print(
            f"ERROR: Expected ID columns ('{ID_COL_FOR_LIPID_DATA}', '{ID_COL_FOR_GENOMIC_DATA}') not in {sample_id_mapping_file}."
        )
        print(f"Available columns: {id_map_df.columns.tolist()}")
        print(
            "Proceeding without ID mapping from this file, which might lead to incorrect sample matching."
        )
        id_map_df = None  # Invalidate map if columns are missing
except FileNotFoundError:
    print(
        f"ERROR: Sample ID mapping file not found: {sample_id_mapping_file}. ID mapping will be problematic."
    )
    id_map_df = None
except Exception as e:
    print(f"ERROR loading sample ID mapping file '{sample_id_mapping_file}': {e}")
    id_map_df = None

# Merge lipid data with mapping data
# The index of merged_lipid_df is 'SampleID_Lipid'
merged_lipid_df.reset_index(inplace=True)  # Make 'SampleID_Lipid' a column for merging

if id_map_df is not None:
    pheno_mapped_df = pd.merge(
        merged_lipid_df,
        id_map_df[[ID_COL_FOR_LIPID_DATA, ID_COL_FOR_GENOMIC_DATA]],
        left_on="SampleID_Lipid",
        right_on=ID_COL_FOR_LIPID_DATA,
        how="inner",
    )
    pheno_mapped_df.rename(columns={ID_COL_FOR_GENOMIC_DATA: "IID"}, inplace=True)
    if pheno_mapped_df.empty:
        print(
            "CRITICAL ERROR: No samples common between lipid data and ID mapping file based on provided columns. Check ID column names and file contents."
        )
        exit()
    print(f"After merging with ID map: {pheno_mapped_df.shape[0]} samples remain.")
else:
    print(
        "WARNING: ID mapping file not used or failed to load. Attempting to use lipid SampleIDs directly as IIDs."
    )
    print(
        "This is unlikely to match genomic data unless lipid SampleIDs are identical to FAM IIDs."
    )
    pheno_mapped_df = merged_lipid_df.copy()
    pheno_mapped_df.rename(columns={"SampleID_Lipid": "IID"}, inplace=True)

pheno_mapped_df["FID"] = pheno_mapped_df[
    "IID"
]  # PLINK convention if no family structure

# --- 3. Prepare Phenotype Files for PLINK (per ethnic group & merged) ---
all_pheno_for_plink_list = []
print("\nPreparing phenotype files for PLINK...")

for ethnic_group, info in lipid_files_info.items():
    fam_file_path = info["fam"]
    group_pheno_df = pheno_mapped_df[pheno_mapped_df["ethnic_group"] == ethnic_group]

    if group_pheno_df.empty:
        print(f"No phenotype data for {ethnic_group} after ID mapping. Skipping.")
        continue

    try:
        fam_df = pd.read_csv(
            fam_file_path,
            sep="\s+",
            header=None,
            usecols=[0, 1],
            names=["FID_fam", "IID_fam"],
            dtype=str,
        )
        # Crucial: Ensure IIDs are strings for merging and that FID/IID from FAM is used for final output FID/IID
        group_pheno_df["IID"] = group_pheno_df["IID"].astype(
            str
        )  # Ensure IID from pheno is string

        # Merge with FAM to get FID_fam and ensure only samples in FAM are included
        final_group_pheno_df = pd.merge(
            group_pheno_df, fam_df, left_on="IID", right_on="IID_fam", how="inner"
        )

        if final_group_pheno_df.empty:
            print(
                f"No common samples between mapped phenotypes and FAM file for {ethnic_group}. Skipping."
            )
            continue

        # Use FID from FAM file, and the scaled phenotype
        plink_output_df = final_group_pheno_df[
            ["FID_fam", "IID", scaled_phenotype_col_name]
        ]
        plink_output_df.columns = [
            "FID",
            "IID",
            TARGET_PHENOTYPE_COL,
        ]  # Use consistent pheno name for PLINK

        output_path = os.path.join(output_dir, f"{ethnic_group.lower()}_pheno.txt")
        plink_output_df.to_csv(output_path, sep="\t", index=False, header=True)
        print(
            f"Saved PLINK phenotype file for {ethnic_group} to {output_path} ({plink_output_df.shape[0]} samples)"
        )
        all_pheno_for_plink_list.append(plink_output_df)
    except FileNotFoundError:
        print(f"ERROR: FAM file not found for {ethnic_group}: {fam_file_path}")
    except Exception as e:
        print(f"ERROR processing or writing phenotype for {ethnic_group}: {e}")

if all_pheno_for_plink_list:
    combined_pheno_for_plink = pd.concat(all_pheno_for_plink_list).drop_duplicates(
        subset=["FID", "IID"]
    )
    combined_output_path = os.path.join(output_dir, "merged_pheno.txt")
    combined_pheno_for_plink.to_csv(
        combined_output_path, sep="\t", index=False, header=True
    )
    print(
        f"\nSaved combined PLINK phenotype file to {combined_output_path} ({combined_pheno_for_plink.shape[0]} unique samples)"
    )
else:
    print(
        "CRITICAL: No per-ethnic group phenotype files were generated. Cannot create combined file."
    )


# --- 4. Prepare Covariate File (Placeholder - to be populated with PCs in script 02) ---
# This file will initially just have FID and IID from the combined phenotype data.
# Later, PCs (and potentially Age/Sex from questionnaire) will be added.
if all_pheno_for_plink_list:  # If combined_pheno_for_plink was created
    covariate_df = combined_pheno_for_plink[["FID", "IID"]].copy()
    covariate_output_path = os.path.join(output_dir, "covariates.txt")
    covariate_df.to_csv(covariate_output_path, sep="\t", index=False, header=True)
    print(f"Initialized covariate file (FID, IID) at {covariate_output_path}")
else:
    print(
        "Cannot initialize covariate file as no combined phenotype data is available."
    )


if (
    "pheno_mapped_df" in locals()
    and not pheno_mapped_df.empty
    and "FID" in pheno_mapped_df.columns
    and "IID" in pheno_mapped_df.columns
    and "ethnic_group" in pheno_mapped_df.columns
):

    sample_info_for_plots_df = (
        pheno_mapped_df[["FID", "IID", "ethnic_group"]].drop_duplicates().copy()
    )
    sample_info_output_path = os.path.join(output_dir, "sample_info_with_ethnicity.txt")
    sample_info_for_plots_df.to_csv(
        sample_info_output_path, sep="\t", index=False, header=True
    )
    print(
        f"\nSaved sample information with ethnicity to {sample_info_output_path} ({sample_info_for_plots_df.shape[0]} unique samples)"
    )
else:
    print("\nWARNING: Could not save dedicated sample information file with ethnicity.")
    if "pheno_mapped_df" not in locals():
        print("         Reason: 'pheno_mapped_df' not found or not created.")
    elif pheno_mapped_df.empty:
        print("         Reason: 'pheno_mapped_df' is empty.")
    else:
        print(
            f"         Reason: One or more key columns (FID, IID, ethnic_group) are missing in pheno_mapped_df. Available columns: {pheno_mapped_df.columns.tolist()}"
        )


print(f"\n--- Data Preparation Script Finished ---")
print(
    f"Target phenotype for PLINK: '{TARGET_PHENOTYPE_COL}' (derived from lipid '{TARGET_LIPID_NAME_FROM_FILE}')"
)
print(
    f"Scaled column name: '{scaled_phenotype_col_name}' (this is what's in the pheno files, but PLINK uses header name)"
)
