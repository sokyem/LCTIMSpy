
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from  CID_tims_analysis import extract_mobilograms

# Attempt to import Annotator from statannot or statannotations.
try:
    from statannot import Annotator
except ImportError:
    try:
        from statannotations.Annotator import Annotator
    except ImportError:
        raise ImportError("Annotator could not be imported from 'statannot' or 'statannotations'. "
                          "Please install one of these packages.")

def extract_and_calculate_mobility_difference(df, target_mzs, rt_ranges, mz_tolerance=0.1, 
                                              apply_smoothing=False, sigma=1,
                                              baseline_correction=True,
                                              intensity_threshold=100):
    """
    For each target m/z value, extract mobilogram data over two retention time ranges and compute mobility differences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'retention_time', 'mz', 'intensity', and 'mobility'.
    target_mzs : list of float
        Target m/z values.
    rt_ranges : list of tuple
        Two retention time ranges, e.g. [(20, 21), (23, 25)].
    mz_tolerance : float, optional
        Tolerance for m/z matching (default: 0.1).
    apply_smoothing : bool, optional
        If True, apply Gaussian smoothing (default: False).
    sigma : float, optional
        Smoothing sigma (default: 1).
    baseline_correction : bool, optional
        If True, perform baseline correction (default: True).
    intensity_threshold : float, optional
        Minimum maximum intensity required per RT range (default: 100).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            target_mz, avg_mobility_rt_range1, avg_mobility_rt_range2, mobility_difference,
            peak_mobility_rt_range1, peak_mobility_rt_range2, peak_mobility_difference.
    """
    if len(rt_ranges) != 2:
        raise ValueError("rt_ranges must be a list of exactly two tuples, e.g. [(start1, end1), (start2, end2)].")
    rt_range1_str = f"{rt_ranges[0][0]}-{rt_ranges[0][1]}"
    rt_range2_str = f"{rt_ranges[1][0]}-{rt_ranges[1][1]}"
    results = []
    for target in target_mzs:
        mobilogram_df = extract_mobilograms(
            df, target, rt_ranges, mz_tolerance, apply_smoothing, sigma,
            baseline_correction=baseline_correction
        )
        df1 = mobilogram_df[mobilogram_df['rt_range'] == rt_range1_str]
        df2 = mobilogram_df[mobilogram_df['rt_range'] == rt_range2_str]
        if df1.empty or df2.empty:
            print(f"Warning: No data for target m/z {target} in one or both RT ranges.")
            continue
        if intensity_threshold is not None:
            max_intensity1 = df1['intensity'].max()
            max_intensity2 = df2['intensity'].max()
            if max_intensity1 < intensity_threshold or max_intensity2 < intensity_threshold:
                print(f"Warning: Target m/z {target} removed due to low max intensity (RT1: {max_intensity1}, RT2: {max_intensity2}).")
                continue
        avg_mobility1 = np.average(df1['mobility'], weights=df1['intensity'])
        avg_mobility2 = np.average(df2['mobility'], weights=df2['intensity'])
        mobility_diff = abs(avg_mobility2 - avg_mobility1)
        peak_mobility1 = df1.loc[df1['intensity'].idxmax(), 'mobility']
        peak_mobility2 = df2.loc[df2['intensity'].idxmax(), 'mobility']
        peak_mobility_diff = abs(peak_mobility2 - peak_mobility1)
        results.append({
            "target_mz": target,
            "avg_mobility_rt_range1": avg_mobility1,
            "avg_mobility_rt_range2": avg_mobility2,
            "mobility_difference": mobility_diff,
            "peak_mobility_rt_range1": peak_mobility1,
            "peak_mobility_rt_range2": peak_mobility2,
            "peak_mobility_difference": peak_mobility_diff
        })
    return pd.DataFrame(results)


def extract_and_calculate_mobility_difference_multi(df_list, target_mzs, rt_ranges, 
                                                     mz_tolerance=0.1, apply_smoothing=False, 
                                                     sigma=1, baseline_correction=True,
                                                     intensity_threshold=100):
    """
    For each sample in df_list (or each value in a dict) and each target m/z, extract mobilogram data
    over two RT ranges and calculate mobility differences using extract_and_calculate_mobility_difference.

    Parameters
    ----------
    df_list : list or dict of pd.DataFrame
        LC–MS DataFrames. If dict, keys are used as sample identifiers.
    target_mzs : list of float
        Target m/z values.
    rt_ranges : list of tuple
        Two retention time ranges, e.g. [(20, 21), (23, 25)].
    mz_tolerance : float, optional
        m/z matching tolerance (default: 0.1).
    apply_smoothing : bool, optional
        If True, apply Gaussian smoothing (default: False).
    sigma : float, optional
        Smoothing sigma (default: 1).
    baseline_correction : bool, optional
        If True, perform baseline correction (default: True).
    intensity_threshold : float, optional
        Minimum maximum intensity per RT range (default: 100).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          sample_id, target_mz, avg_mobility_rt_range1, avg_mobility_rt_range2,
          mobility_difference, peak_mobility_rt_range1, peak_mobility_rt_range2,
          peak_mobility_difference.
    """
    if len(rt_ranges) != 2:
        raise ValueError("rt_ranges must be a list of exactly two tuples, e.g. [(start1, end1), (start2, end2)].")
    if isinstance(df_list, dict):
        sample_ids = list(df_list.keys())
        dfs = list(df_list.values())
    else:
        sample_ids = None
        dfs = df_list
    results = []
    for idx, df in enumerate(dfs):
        sample_id = sample_ids[idx] if sample_ids is not None else idx
        res = extract_and_calculate_mobility_difference(
            df, target_mzs, rt_ranges, mz_tolerance, apply_smoothing, sigma,
            baseline_correction, intensity_threshold
        )
        if not res.empty:
            res["sample_id"] = sample_id
            results.append(res)
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def compare_targetmz_differences_with_boxplot(df_list, target_mzs, rt_ranges, mz_tolerance=0.1, 
                                              apply_smoothing=False, sigma=1,
                                              intensity_threshold=None,
                                              baseline_correction=True,
                                              metric='mobility_difference',
                                              apply_correction=True,
                                              correction_method='bonferroni',
                                              alpha=0.05,
                                              df_name='DataFrame',
                                              save_figure=False,
                                              file_name='plot',
                                              file_format='png',
                                              dpi=300,
                                              box_width=0.5,
                                              **point_plot_kwargs):
    """
    Compare mobility differences for target m/z values across replicate samples using a box plot.
    
    The function:
      1. Extracts mobility difference data using extract_and_calculate_mobility_difference_multi.
      2. Removes target m/z with data from fewer than three distinct samples.
      3. Melts the DataFrame for seaborn plotting.
      4. Performs pairwise Welch t-tests and applies multiple testing correction.
      5. Generates a box plot with statistical annotations for significant comparisons.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        List of LC–MS DataFrames.
    target_mzs : list of float
        Target m/z values.
    rt_ranges : list of tuple
        Two retention time ranges, e.g. [(20,21), (23,25)].
    mz_tolerance : float, optional
        m/z tolerance (default: 0.1).
    apply_smoothing : bool, optional
        Apply Gaussian smoothing (default: False).
    sigma : float, optional
        Smoothing sigma (default: 1).
    intensity_threshold : float or None, optional
        If provided, targets with max intensity below this in either RT range are skipped.
    baseline_correction : bool, optional
        Subtract minimum intensity from each RT range (default: True).
    metric : str, optional
        Mobility metric to compare (default: 'mobility_difference').
    apply_correction : bool, optional
        Apply multiple testing correction (default: True).
    correction_method : str, optional
        Correction method: 'bonferroni', 'holm', 'tukey', or 'none' (default: 'bonferroni').
    alpha : float, optional
        Significance level (default: 0.05).
    df_name : str, optional
        Name for the DataFrame (used in title).
    save_figure : bool, optional
        If True, save the figure to disk.
    file_name : str, optional
        File name for saving (default: 'plot').
    file_format : str, optional
        File format (default: 'png').
    dpi : int, optional
        Resolution (default: 300).
    box_width : float, optional
        Width of boxes in the plot (default: 0.5).
    point_plot_kwargs : dict, optional
        Additional keyword arguments for swarm plot overlay.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, matplotlib.figure.Figure)
        Mobility difference DataFrame, t-test summary DataFrame, and the box plot figure.
    """
    mobility_df = extract_and_calculate_mobility_difference_multi(
        df_list, target_mzs, rt_ranges, mz_tolerance, 
        apply_smoothing, sigma, intensity_threshold=intensity_threshold, 
        baseline_correction=baseline_correction
    )
    sample_counts = mobility_df.groupby('target_mz')['sample_id'].nunique()
    valid_targets = sample_counts[sample_counts >= 3].index.tolist()
    if not valid_targets:
        raise ValueError("No target m/z has data in at least three samples for pairwise analysis.")
    mobility_df = mobility_df[mobility_df['target_mz'].isin(valid_targets)]
    df_melted = mobility_df.copy().rename(columns={metric: 'Mobility_Diff'})
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='target_mz', y='Mobility_Diff', data=df_melted, palette="Set3", width=box_width)
    sns.swarmplot(x='target_mz', y='Mobility_Diff', data=df_melted, color=".25", ax=ax, **point_plot_kwargs)
    target_values = df_melted['target_mz'].unique()
    pairs = [(a, b) for idx, a in enumerate(target_values) for b in target_values[idx+1:]]
    p_values = []
    for (group1, group2) in pairs:
        data1 = df_melted[df_melted['target_mz'] == group1]['Mobility_Diff']
        data2 = df_melted[df_melted['target_mz'] == group2]['Mobility_Diff']
        stat, p_val = ttest_ind(data1, data2, nan_policy='omit')
        p_values.append(p_val)
    if apply_correction:
        if correction_method in ['bonferroni', 'holm']:
            reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
        elif correction_method == 'tukey':
            tukey_data = df_melted[['target_mz', 'Mobility_Diff']].dropna()
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey_result = pairwise_tukeyhsd(endog=tukey_data['Mobility_Diff'],
                                             groups=tukey_data['target_mz'], alpha=alpha)
            reject = tukey_result.reject
            corrected_p_values = tukey_result.pvalues
        else:
            raise ValueError(f"Unsupported correction method: {correction_method}")
    else:
        corrected_p_values = p_values
        reject = np.array(corrected_p_values) < alpha
    significant_pairs = [pair for pair, sig in zip(pairs, reject) if sig]
    significant_p_values = [p for p, sig in zip(corrected_p_values, reject) if sig]
    if significant_pairs:
        annotator = Annotator(ax, significant_pairs, x='target_mz', y='Mobility_Diff', data=df_melted)
        annotator.configure(test=None, text_format='star', loc='inside', verbose=2,
                            pvalue_thresholds=[(1e-4, '****'), (1e-3, '***'), (1e-2, '**'), (0.05, '*')])
        annotator.set_pvalues_and_annotate(significant_p_values)
    ax.set_xlabel('Target m/z')
    ax.set_ylabel('Mobility Difference')
    ax.set_title(f"Pairwise Comparison for: {df_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    if save_figure:
        valid_formats = ['png', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff', 'bmp', 'eps']
        if file_format not in valid_formats:
            raise ValueError(f"Invalid file format: '{file_format}'. Supported formats are: {', '.join(valid_formats)}.")
        plt.savefig(f"{file_name}.{file_format}", format=file_format, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    ttest_df = pd.DataFrame({
        'pairs': pairs,
        'raw_p_value': p_values,
        'adjusted_p_value': corrected_p_values,
        'reject': reject
    })
    return mobility_df, ttest_df, ax.get_figure()