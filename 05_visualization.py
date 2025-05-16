# 05_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats


# --- Configuration ---
gwas_results_dir = "gwas_results"
visualization_output_dir = "visualization_plots"
os.makedirs(visualization_output_dir, exist_ok=True)

# Define GWAS results to plot (prefix, title)
# These should match the output prefixes from 04_gwas_on_selected.py
# The .qassoc or .assoc.linear file will be used. PLINK --assoc outputs .qassoc for quantitative.
# If --linear was used, it would be .assoc.linear
# Let's assume .qassoc from '--assoc'
gwas_files_to_plot = {
    "ML_Selected_Merged": {
        "file_prefix": os.path.join(gwas_results_dir, "gwas_ml_selected_merged"),
        "title_suffix": "ML Selected (Merged)",
        "map_file_prefix": os.path.join(
            "pca_data", "merged_for_pca_unpruned"
        ),  # BIM for CHR/BP for these SNPs
    },
    "Traditional_Merged_LDpruned": {
        "file_prefix": os.path.join(
            gwas_results_dir, "gwas_traditional_merged_ldpruned"
        ),
        "title_suffix": "Traditional LD-Pruned (Merged)",
        "map_file_prefix": os.path.join(
            "pca_data", "merged_for_pca_ldpruned_final"
        ),  # BIM for these SNPs
    },
    # Add per-population GWAS results if desired
    # 'ML_Selected_Chinese': {
    #     'file_prefix': os.path.join(gwas_results_dir, 'gwas_ml_selected_chinese'),
    #     'title_suffix': 'ML Selected (Chinese)',
    #     'map_file_prefix': os.path.join('qc_data', 'chinese_s2_mind') # BIM for CHR/BP for these SNPs
    # },
}
# Add Chinese, Malay, Indian if per-population GWAS was run and you want to plot them
for pop_lower in ["chinese", "malay", "indian"]:
    key_name = f"ML_Selected_{pop_lower.capitalize()}"
    file_prefix_val = os.path.join(gwas_results_dir, f"gwas_ml_selected_{pop_lower}")
    # The BIM file should correspond to the bfile used for this GWAS run
    map_file_prefix_val = os.path.join("qc_data", f"{pop_lower}_s2_mind")

    # Check if the primary result file exists before adding to dict
    if os.path.exists(f"{file_prefix_val}.qassoc"):
        gwas_files_to_plot[key_name] = {
            "file_prefix": file_prefix_val,
            "title_suffix": f"ML Selected ({pop_lower.capitalize()})",
            "map_file_prefix": map_file_prefix_val,
        }


SIGNIFICANCE_THRESHOLD_BONF = 5e-8  # Common Bonferroni threshold
SIGNIFICANCE_THRESHOLD_SUGG = 1e-5  # Suggestive line


def load_bim_file(bim_file_path):
    """Loads SNP, CHR, BP from a BIM file."""
    try:
        bim_df = pd.read_csv(
            bim_file_path,
            sep="\s+",
            header=None,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
            dtype={"CHR": str, "SNP": str, "BP": int},
        )
        return bim_df[["SNP", "CHR", "BP"]]
    except FileNotFoundError:
        print(f"BIM file not found: {bim_file_path}")
    except Exception as e:
        print(f"Error loading BIM {bim_file_path}: {e}")
    return pd.DataFrame(columns=["SNP", "CHR", "BP"])


def manhattan_plot(
    gwas_results_df, bim_df, title="Manhattan Plot", bonf_thresh=5e-8, sugg_thresh=1e-5
):
    df = pd.merge(gwas_results_df[["SNP", "P"]], bim_df, on="SNP", how="inner")
    df = df.dropna(subset=["P", "CHR", "BP"])
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"])
    df = df[df["P"] > 0]  # Ensure P-values are positive
    if df.empty:
        print(f"No data to plot for {title} after filtering.")
        return None

    df["-log10P"] = -np.log10(df["P"])

    df["CHR"] = df["CHR"].astype(str)
    # Handle X, Y, MT chromosomes by converting to numeric (e.g., 23, 24, 25)
    chr_map = {
        **{str(i): i for i in range(1, 23)},
        "X": 23,
        "Y": 24,
        "MT": 25,
        "XY": 25,
    }  # XY if present
    df["CHR_numeric"] = (
        df["CHR"].apply(lambda x: chr_map.get(x, pd.NA)).astype("float").astype("Int64")
    )  # Allow NA then Int
    df = df.dropna(subset=["CHR_numeric"])
    df = df.sort_values(["CHR_numeric", "BP"])

    df["ind"] = range(len(df))
    df_grouped = df.groupby("CHR_numeric")

    colors = ["#0072B2", "#D55E00"]  # Two alternating colors for chromosomes
    x_labels = []
    x_labels_pos = []

    plt.figure(figsize=(18, 7))
    current_pos_offset = 0
    last_max_pos = 0

    df["plot_pos"] = 0  # Initialize plot_pos column

    for i, (name, group) in enumerate(df.groupby("CHR_numeric")):
        if i > 0:  # Add spacing between chromosomes
            current_pos_offset = last_max_pos + (
                group["BP"].max() / 10
            )  # Arbitrary spacing based on previous chr size

        group_plot_pos = group["BP"] + current_pos_offset
        df.loc[group.index, "plot_pos"] = group_plot_pos  # Assign calculated plot_pos
        last_max_pos = group_plot_pos.max()  # Update last_max_pos for next iteration

        plt.scatter(
            group_plot_pos,
            group["-log10P"],
            color=colors[i % len(colors)],
            s=15,
            alpha=0.7,
        )
        x_labels.append(df.loc[group.index, "CHR"].iloc[0])  # Get original CHR label
        x_labels_pos.append(group_plot_pos.mean())

    if bonf_thresh:
        plt.axhline(
            y=-np.log10(bonf_thresh),
            color="r",
            linestyle="--",
            lw=1.2,
            label=f"Bonferroni ({bonf_thresh:.0e})",
        )
    if sugg_thresh:
        plt.axhline(
            y=-np.log10(sugg_thresh),
            color="grey",
            linestyle=":",
            lw=1.2,
            label=f"Suggestive ({sugg_thresh:.0e})",
        )

    if x_labels_pos:  # Only set ticks if there are labels
        plt.xticks(x_labels_pos, x_labels, rotation=45, ha="right")
    else:
        plt.xticks([])  # No ticks if no data plotted

    plt.xlabel("Chromosome")
    plt.ylabel(r"$-\log_{10}(P)$")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    return plt


def qq_plot(p_values, title="QQ Plot"):
    p_values = p_values.dropna()
    p_values = p_values[(p_values > 0) & (p_values <= 1)]
    if len(p_values) == 0:
        print(f"No valid p-values for QQ plot: {title}")
        return None

    observed_p = -np.log10(np.sort(p_values))
    expected_p = -np.log10(
        np.arange(1, len(observed_p) + 1) / (len(observed_p) + 1.0)
    )  # Add 1 to denominator

    plt.figure(figsize=(7, 7))
    plt.scatter(expected_p, observed_p, s=10, c="#0072B2", alpha=0.7)
    plt.plot(
        [0, max(expected_p.max(), observed_p.max())],
        [0, max(expected_p.max(), observed_p.max())],
        color="r",
        linestyle="-",
        lw=1.5,
        label="Expected under null",
    )
    plt.xlabel(r"Expected $-\log_{10}(P)$")
    plt.ylabel(r"Observed $-\log_{10}(P)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt


print("\n--- Generating Plots ---")
for key, info in gwas_files_to_plot.items():
    gwas_file = (
        f"{info['file_prefix']}.qassoc"  # Or .assoc.linear depending on PLINK command
    )
    bim_file = f"{info['map_file_prefix']}.bim"
    plot_title_suffix = info["title_suffix"]

    print(f"\nProcessing {plot_title_suffix} (File: {gwas_file})")

    if not os.path.exists(gwas_file):
        print(f"GWAS result file not found: {gwas_file}. Skipping plots.")
        continue
    if not os.path.exists(bim_file):
        print(
            f"BIM file not found: {bim_file}. Needed for Manhattan plot CHR/BP. Skipping Manhattan for {plot_title_suffix}."
        )
        # Still try QQ plot

    try:
        gwas_df = pd.read_csv(gwas_file, sep="\s+")
        if "P" not in gwas_df.columns:
            print(
                f"'P' column not found in {gwas_file}. Available: {gwas_df.columns.tolist()}. Skipping plots."
            )
            continue

        # Manhattan Plot
        if os.path.exists(bim_file):
            bim_data = load_bim_file(bim_file)
            if not bim_data.empty:
                man_plt = manhattan_plot(
                    gwas_df,
                    bim_data,
                    title=f"Manhattan Plot: {plot_title_suffix}",
                    bonf_thresh=SIGNIFICANCE_THRESHOLD_BONF,
                    sugg_thresh=SIGNIFICANCE_THRESHOLD_SUGG,
                )
                if man_plt:
                    man_plot_filename = os.path.join(
                        visualization_output_dir, f"manhattan_{key}.png"
                    )
                    man_plt.savefig(man_plot_filename)
                    print(f"Saved Manhattan plot to {man_plot_filename}")
                    man_plt.close()  # Close plot to free memory
            else:
                print(
                    f"BIM data was empty for {bim_file}. Cannot create Manhattan plot."
                )

        # QQ Plot
        qq_plt = qq_plot(gwas_df["P"], title=f"QQ Plot: {plot_title_suffix}")
        if qq_plt:
            qq_plot_filename = os.path.join(
                visualization_output_dir, f"qqplot_{key}.png"
            )
            qq_plt.savefig(qq_plot_filename)
            print(f"Saved QQ plot to {qq_plot_filename}")
            qq_plt.close()  # Close plot

    except FileNotFoundError:
        print(f"Error: One of the files for {plot_title_suffix} not found.")
    except Exception as e:
        print(f"Error processing plots for {plot_title_suffix}: {e}")


def plot_pca_results(
    eigenvec_file,
    sample_info_file,
    output_filename,
    fid_col_pca="FID",
    iid_col_pca="IID",
    pc1_col_pca="PC1",
    pc2_col_pca="PC2",
    fid_col_info="FID",
    iid_col_info="IID",
    ethnic_group_col_info="ethnic_group",
):
    """
    Generates and saves a PCA plot (PC1 vs PC2) colored by ethnic group.

    Args:
        eigenvec_file (str): Path to the PLINK .eigenvec PCA output file.
        sample_info_file (str): Path to a file containing sample FID, IID, and ethnic group.
                                (e.g., merged_pheno.txt or covariates.txt if ethnic_group is added there)
        output_filename (str): Path to save the generated PCA plot.
        fid_col_pca (str): Column name for FID in the eigenvec file.
        iid_col_pca (str): Column name for IID in the eigenvec file.
        pc1_col_pca (str): Column name for PC1 in the eigenvec file.
        pc2_col_pca (str): Column name for PC2 in the eigenvec file.
        fid_col_info (str): Column name for FID in the sample info file.
        iid_col_info (str): Column name for IID in the sample info file.
        ethnic_group_col_info (str): Column name for ethnic group in the sample info file.
    """
    print(f"\nGenerating PCA plot from {eigenvec_file} and {sample_info_file}")
    try:
        # Load PCA eigenvectors
        pca_df = pd.read_csv(
            eigenvec_file, sep="\s+", dtype={fid_col_pca: str, iid_col_pca: str}
        )
        # Handle potential '#' prefix in PLINK2 eigenvector file header
        if pca_df.columns[0].startswith("#"):
            pca_df = pca_df.rename(
                columns={pca_df.columns[0]: fid_col_pca, pca_df.columns[1]: iid_col_pca}
            )

        # Load sample information with ethnic groups
        # Assuming tab-separated for pheno/covariate files
        sample_info_df = pd.read_csv(
            sample_info_file,
            sep="\t",
            usecols=[fid_col_info, iid_col_info, ethnic_group_col_info],
            dtype={fid_col_info: str, iid_col_info: str, ethnic_group_col_info: str},
        )
        sample_info_df = sample_info_df.drop_duplicates(
            subset=[fid_col_info, iid_col_info]
        )

        # Merge PCA data with ethnic group information
        # Ensure correct merging columns if FID is not always identical to IID or if one is preferred
        # For PLINK output, FID and IID together make a unique key.
        merged_df = pd.merge(
            pca_df[[fid_col_pca, iid_col_pca, pc1_col_pca, pc2_col_pca]],
            sample_info_df,
            left_on=[fid_col_pca, iid_col_pca],
            right_on=[fid_col_info, iid_col_info],
            how="inner",
        )

        if merged_df.empty:
            print(
                "ERROR: No common samples found between PCA eigenvectors and sample info file. Cannot generate PCA plot."
            )
            print(f"PCA file head:\n{pca_df.head()}")
            print(f"Sample Info file head:\n{sample_info_df.head()}")
            return

        # Create the plot
        plt.figure(figsize=(10, 8))
        ethnic_groups = merged_df[ethnic_group_col_info].unique()
        colors = plt.cm.get_cmap(
            "viridis", len(ethnic_groups)
        )  # Or 'tab10', 'Set1' etc.

        for i, group in enumerate(ethnic_groups):
            subset = merged_df[merged_df[ethnic_group_col_info] == group]
            plt.scatter(
                subset[pc1_col_pca],
                subset[pc2_col_pca],
                label=group,
                color=colors(i),
                alpha=0.7,
                s=50,
            )

        plt.title("PCA of iOmics Populations (PC1 vs PC2)")
        plt.xlabel(
            f"{pc1_col_pca} (Explains 13.2298% variance)"
        )  # You'll need to get variance explained if PLINK provides it
        plt.ylabel(f"{pc2_col_pca} (Explains 3.07228% variance)")
        plt.legend(title="Ethnic Group")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved PCA plot to {output_filename}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found during PCA plot generation: {e}")
    except KeyError as e:
        print(
            f"ERROR: Missing expected column during PCA plot generation: {e}. Check column names."
        )
        print(
            "Ensure PCA file has FID, IID, PC1, PC2 and info file has FID, IID, ethnic_group."
        )
    except Exception as e:
        print(f"An error occurred generating the PCA plot: {e}")


# --- (At the end of your 05_visualization.py script, after other plotting calls) ---
# Define paths for PCA plotting
pca_eigenvec_filepath = os.path.join("pca_data", "merged_pca.eigenvec")
# The 'merged_pheno.txt' from script 01 should contain FID, IID, and ethnic_group
# Make sure the 'ethnic_group' column name is correct.
# In script 01, pheno_mapped_df had 'ethnic_group', and this should persist to 'merged_pheno.txt'
sample_info_for_pca_plot = os.path.join(
    "prepared_data", "sample_info_with_ethnicity.txt"
)
pca_plot_output_filename = os.path.join(
    visualization_output_dir, "pca_populations_plot.png"
)

if os.path.exists(pca_eigenvec_filepath) and os.path.exists(sample_info_for_pca_plot):
    plot_pca_results(
        eigenvec_file=pca_eigenvec_filepath,
        sample_info_file=sample_info_for_pca_plot,  # This now points to the new file
        output_filename=pca_plot_output_filename,
        fid_col_pca="FID",
        iid_col_pca="IID",
        pc1_col_pca="PC1",
        pc2_col_pca="PC2",
        fid_col_info="FID",
        iid_col_info="IID",
        ethnic_group_col_info="ethnic_group",  # This should now match the column in the new file
    )
else:
    print(
        "\nSkipping PCA plot generation: PCA eigenvector file or sample info file not found."
    )
    if not os.path.exists(pca_eigenvec_filepath):
        print(f" - Missing: {pca_eigenvec_filepath}")
    if not os.path.exists(sample_info_for_pca_plot):
        print(f" - Missing: {sample_info_for_pca_plot}")


print("\n--- Visualization Script Finished ---")
