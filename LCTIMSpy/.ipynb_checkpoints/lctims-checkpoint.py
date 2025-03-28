#!/usr/bin/env python
"""
Module: timstof_analysis
Description: Functions for analyzing TIMSTOF data.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK import *
from pyTDFSDK.classes import TdfData, CHROMATOGRAM_JOB_GENERATOR
from pyTDFSDK.tsf import tsf_read_line_spectrum_v2, tsf_index_to_mz
from pyTDFSDK.ctypes_data_structures import PressureCompensationStrategy
from pyteomics import mass

# Attempt to import Annotator from statannot or statannotations.
try:
    from statannot import Annotator
except ImportError:
    try:
        from statannotations.Annotator import Annotator
    except ImportError:
        raise ImportError("Annotator could not be imported from 'statannot' or 'statannotations'. "
                          "Please install one of these packages.")

pio.renderers.default = 'notebook'
dll = init_tdf_sdk_api()


def extract_lcms_tdf_data(tdf_data, exclude_mobility=False, mode=None, profile_bins=None,
                          mz_encoding=None, intensity_encoding=None, diapasef_window=None,
                          chunk_size=100, mobility_bin_width=None):
    """
    Extract LC–MS data from a TDF data object in a format similar to Bruker DataAnalysis.

    Parameters
    ----------
    tdf_data : object
        TDF data object.
    exclude_mobility : bool, optional
        If True, mobility data is excluded (default: False).
    mode, profile_bins, mz_encoding, intensity_encoding, diapasef_window : 
        Parameters for 2D extraction if mobility is excluded.
    chunk_size : int, optional
        Number of frames per chunk (default: 100).
    mobility_bin_width : float or None, optional
        If provided, mobility values are binned by rounding to the nearest multiple.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame, polarity, retention_time, mz, intensity, and (if not excluded) mobility.
    """
    frames_df = tdf_data.analysis['Frames']
    frames_map = frames_df.set_index('Id').to_dict(orient='index')
    frames_list = list(frames_map.keys())
    lcms_data_list = []
    n_chunks = max(1, len(frames_list) // chunk_size)
    frame_chunks = np.array_split(frames_list, n_chunks)
    
    for chunk in frame_chunks:
        for frame in chunk:
            try:
                record = frames_map.get(frame)
                if record is None:
                    continue
                if int(record.get('MsMsType', 0)) != 0:
                    continue
                num_scans = int(record.get('NumScans', 0))
                if not exclude_mobility:
                    mz_array, intensity_array, mobility_array = extract_3d_tdf_spectrum(
                        tdf_data, frame, 0, num_scans
                    )
                else:
                    mz_array, intensity_array = extract_2d_tdf_spectrum(
                        tdf_data, frame, 0, num_scans, mode, profile_bins, mz_encoding, intensity_encoding
                    )
                    mobility_array = None
                if mz_array is not None and intensity_array is not None:
                    n_points = len(mz_array)
                    frame_data = {
                        'frame': [frame] * n_points,
                        'polarity': [record.get('Polarity')] * n_points,
                        'retention_time': [float(record.get('Time', 0)) / 60] * n_points,
                        'mz': mz_array,
                        'intensity': intensity_array
                    }
                    if not exclude_mobility and mobility_array is not None:
                        if mobility_bin_width is not None:
                            mobility_array = np.array(mobility_array)
                            mobility_binned = np.round(mobility_array / mobility_bin_width) * mobility_bin_width
                            frame_data['mobility'] = mobility_binned
                        else:
                            frame_data['mobility'] = mobility_array
                    lcms_data_list.append(pd.DataFrame(frame_data))
            except Exception as e:
                print(f"Skipping frame {frame} due to error: {e}")
    return pd.concat(lcms_data_list, ignore_index=True) if lcms_data_list else pd.DataFrame()


def extract_chromatograms(df, target_mzs, mz_tolerance=0.01):
    """
    Extract chromatograms (EICs) from the input DataFrame for each target m/z value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns: 'frame', 'polarity', 'retention_time', 'mz', 
        'intensity', and 'mobility'.
    target_mzs : float or list of float
        A target m/z value or a list of target m/z values.
    mz_tolerance : float, optional
        Tolerance for m/z matching (default: 0.01).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'target_mz', 'retention_time', 'intensity'. Each row represents
        the summed intensity at a given retention time for a target m/z.
    """
    if not isinstance(target_mzs, (list, np.ndarray)):
        target_mzs = [target_mzs]
    df_sorted = df.sort_values('mz').reset_index(drop=True)
    mz_values = df_sorted['mz'].values
    results = []
    for target in target_mzs:
        left_idx = np.searchsorted(mz_values, target - mz_tolerance, side='left')
        right_idx = np.searchsorted(mz_values, target + mz_tolerance, side='right')
        df_target = df_sorted.iloc[left_idx:right_idx]
        if df_target.empty:
            continue
        df_target = df_target[np.abs(df_target['mz'] - target) <= mz_tolerance]
        if df_target.empty:
            continue
        grouped = df_target.groupby('retention_time', as_index=False, sort=False)['intensity'].sum()
        grouped['target_mz'] = target
        grouped = grouped.sort_values('retention_time')
        results.append(grouped)
    if results:
        result_df = pd.concat(results, ignore_index=True)
        result_df = result_df[['target_mz', 'retention_time', 'intensity']]
        return result_df
    else:
        return pd.DataFrame(columns=['target_mz', 'retention_time', 'intensity'])





def extract_mobilogram(df, target_mz, rt_ranges, mz_tolerance=0.1, 
                       apply_smoothing=False, sigma=1, 
                       baseline_correction=True):
    """
    Extract mobilogram data for a given target m/z across specified retention time ranges,
    mimicking Bruker DataAnalysis. Groups data by exact mobility and applies baseline correction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'retention_time', 'mz', 'intensity', 'mobility'.
    target_mz : float
        The target m/z value.
    rt_ranges : list of tuple
        Retention time ranges as tuples, e.g. [(start_rt1, end_rt1), (start_rt2, end_rt2)].
    mz_tolerance : float, optional
        Tolerance for m/z matching (default: 0.1).
    apply_smoothing : bool, optional
        If True, apply Gaussian smoothing to intensity values (default: False).
    sigma : float, optional
        Standard deviation for smoothing (default: 1).
    baseline_correction : bool, optional
        If True, subtract the minimum intensity from each group (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'mobility', 'intensity', 'rt_range'.
    """
    df['mz'] = pd.to_numeric(df['mz'], errors='coerce')
    mobilogram_frames = []
    df_filtered = df[np.abs(df['mz'] - target_mz) <= mz_tolerance].copy()
    for start_rt, end_rt in rt_ranges:
        df_rt = df_filtered[(df_filtered['retention_time'] >= start_rt) & 
                            (df_filtered['retention_time'] <= end_rt)]
        if df_rt.empty:
            continue
        mob_group = df_rt.groupby('mobility', as_index=False)['intensity'].sum()
        if apply_smoothing and not mob_group.empty:
            mob_group['intensity'] = gaussian_filter1d(mob_group['intensity'].values, sigma=sigma)
        if baseline_correction and not mob_group.empty:
            mob_group['intensity'] = mob_group['intensity'] - mob_group['intensity'].min()
        mob_group['rt_range'] = f"{start_rt}-{end_rt}"
        mobilogram_frames.append(mob_group)
    if mobilogram_frames:
        result_df = pd.concat(mobilogram_frames, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=['mobility', 'intensity', 'rt_range'])
    return result_df


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
        mobilogram_df = extract_mobilogram_df(
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


def normalize_series(x, norm_range):
    """
    Normalize a pandas Series to a specified range.

    Parameters
    ----------
    x : pd.Series
        Series to normalize.
    norm_range : tuple
        Desired output range (min, max).

    Returns
    -------
    np.array
        Normalized values.
    """
    x = x.copy().fillna(0)
    if x.max() == x.min():
        return np.full(x.shape, norm_range[0])
    return (x - x.min()) / (x.max() - x.min()) * (norm_range[1] - norm_range[0]) + norm_range[0]


def plot_mobilogram_target_mz(df, target_mzs, rt_ranges, mz_tolerance=0.1, 
                              apply_smoothing=False, sigma=1, 
                              normalize=True, normalize_range=(0, 1),
                              overlay=False, save_plots=False, save_directory="plots",
                              baseline_correction=True):
    """
    Extract mobilogram data for target m/z values using extract_mobilogram_df and plot mobility vs. intensity.
    Forces first and last intensity values to zero per RT range (to yield a Gaussian-like baseline).

    Parameters
    ----------
    df : pd.DataFrame or list of pd.DataFrame
        DataFrame(s) with columns: 'retention_time', 'mz', 'intensity', 'mobility'.
    target_mzs : float or list of float
        Target m/z value(s).
    rt_ranges : list of tuple
        Retention time ranges as (start, end).
    mz_tolerance : float, optional
        m/z tolerance (default: 0.1).
    apply_smoothing : bool, optional
        If True, apply Gaussian smoothing (default: False).
    sigma : float, optional
        Smoothing sigma (default: 1).
    normalize : bool, optional
        If True, normalize intensities to normalize_range (default: True).
    normalize_range : tuple, optional
        Normalization range (default: (0, 1)).
    overlay : bool, optional
        If True, overlay all target traces in one figure per dataset; otherwise, separate figures.
    save_plots : bool, optional
        If True, save figures to disk.
    save_directory : str, optional
        Directory to save figures.
    baseline_correction : bool, optional
        If True, subtract the minimum intensity from each RT range and force endpoints to zero.

    Returns
    -------
    list of matplotlib.figure.Figure
        List of generated figures.
    """
    if isinstance(df, list):
        dataset_names = [getattr(d, 'name', f"Dataset {i+1}") for i, d in enumerate(df)]
        combined_dataset_name = ", ".join(dataset_names)
    else:
        combined_dataset_name = getattr(df, 'name', "Dataset 1")
    if not isinstance(target_mzs, (list, tuple, np.ndarray)):
        target_mzs = [target_mzs]
    if save_plots and not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    figures = []
    if overlay:
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.viridis
        n_targets = len(target_mzs)
        target_colors = {target: cmap(i / max(1, n_targets - 1)) for i, target in enumerate(target_mzs)}
        for target in target_mzs:
            mob_df = extract_mobilogram_df(
                df, target, rt_ranges, mz_tolerance, apply_smoothing, sigma,
                baseline_correction=baseline_correction
            )
            if mob_df.empty:
                print(f"No mobilogram data for target m/z {target} in {combined_dataset_name}")
                continue
            if normalize:
                mob_df['normalized_intensity'] = mob_df.groupby('rt_range')['intensity']\
                                                       .transform(lambda x: normalize_series(x, normalize_range))
            else:
                mob_df['normalized_intensity'] = mob_df['intensity']
            for rt in sorted(mob_df['rt_range'].unique()):
                subset = mob_df[mob_df['rt_range'] == rt].sort_values('mobility')
                if not subset.empty:
                    subset.loc[subset.index[0], 'normalized_intensity'] = 0
                    subset.loc[subset.index[-1], 'normalized_intensity'] = 0
                    ax.plot(
                        subset['mobility'],
                        subset['normalized_intensity'],
                        linestyle='-',
                        label=f"m/z {target}" if rt == sorted(mob_df['rt_range'].unique())[0] else None,
                        color=target_colors[target]
                    )
        ax.set_xlabel("Mobility")
        ax.set_ylabel("Normalized Intensity" if normalize else "Intensity")
        ax.set_title(f"Overlayed Mobilograms for {combined_dataset_name}" + (" (Normalized)" if normalize else ""))
        ax.legend(title="Target m/z", fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        fig.subplots_adjust(right=0.75)
        if save_plots:
            filename = os.path.join(save_directory, "overlayed_mobilograms.svg")
            plt.rcParams['svg.fonttype'] = 'none'
            fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        figures.append(fig)
    else:
        figs = []
        unique_rt_ranges = sorted([f"{start}-{end}" for start, end in rt_ranges])
        n_ranges = len(unique_rt_ranges)
        cmap = plt.cm.viridis
        rt_colors = {rt: cmap(i / max(1, n_ranges - 1)) for i, rt in enumerate(unique_rt_ranges)}
        for target in target_mzs:
            fig, ax = plt.subplots(figsize=(10, 6))
            mob_df = extract_mobilogram_df(
                df, target, rt_ranges, mz_tolerance, apply_smoothing, sigma,
                baseline_correction=baseline_correction
            )
            if mob_df.empty:
                print(f"No mobilogram data for target m/z {target} in {combined_dataset_name}")
                plt.close(fig)
                continue
            if normalize:
                mob_df['normalized_intensity'] = mob_df.groupby('rt_range')['intensity']\
                                                       .transform(lambda x: normalize_series(x, normalize_range))
            else:
                mob_df['normalized_intensity'] = mob_df['intensity']
            for rt in sorted(mob_df['rt_range'].unique()):
                subset = mob_df[mob_df['rt_range'] == rt].sort_values('mobility')
                if not subset.empty:
                    subset.loc[subset.index[0], 'normalized_intensity'] = 0
                    subset.loc[subset.index[-1], 'normalized_intensity'] = 0
                    ax.plot(
                        subset['mobility'],
                        subset['normalized_intensity'],
                        linestyle='-',
                        label=f"RT {rt}",
                        color=rt_colors[rt]
                    )
            ax.set_xlabel("Mobility")
            ax.set_ylabel("Normalized Intensity" if normalize else "Intensity")
            ax.set_title(f"Mobilogram for {combined_dataset_name} - Target m/z: {target}" + (" (Normalized)" if normalize else ""))
            ax.legend(title="RT Range", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            fig.subplots_adjust(right=0.75)
            if save_plots:
                filename = os.path.join(save_directory, f"target_{target}_mobilogram.svg")
                plt.rcParams['svg.fonttype'] = 'none'
                fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
            plt.show()
            figs.append(fig)
    return figures


def plot_interactive_chromatograms(df, target_mzs, rt_ranges, mz_tolerance=0.1, 
                                   apply_smoothing=False, sigma=1, 
                                   normalize=True, normalize_range=(0, 1),
                                   overlay=False, save_plots=False, save_directory="plots",
                                   baseline_correction=True,
                                   major_ticks=None, minor_ticks=None,
                                   max_points=None):
    """
    Create interactive Plotly chromatogram plots (EICs) for specified target m/z values.
    Uses WebGL (Scattergl) for performance and scales y-values for readability.
    Optionally, intensities are normalized per RT range and endpoints forced to zero.
    Downsampling is applied if the number of points exceeds max_points.

    Parameters
    ----------
    df : pd.DataFrame, list of pd.DataFrame, or dict
        DataFrame(s) with columns: 'retention_time', 'mz', 'intensity', 'mobility'.
        If a dict, keys are used as dataset names.
    target_mzs : float or list of float
        Target m/z value(s).
    rt_ranges : list of tuple
        Retention time ranges as (rt_min, rt_max).
    mz_tolerance : float, optional
        m/z tolerance (default: 0.1).
    apply_smoothing : bool, optional
        Apply Gaussian smoothing (default: False).
    sigma : float, optional
        Smoothing sigma (default: 1).
    normalize : bool, optional
        Normalize intensities to normalize_range (default: True).
    normalize_range : tuple, optional
        Normalization range (default: (0, 1)).
    overlay : bool, optional
        If True, create one interactive figure per dataset with all target traces overlaid;
        if False, separate figures per target m/z.
    save_plots : bool, optional
        If True, save each figure as an HTML file.
    save_directory : str, optional
        Directory to save figures.
    baseline_correction : bool, optional
        If True, subtract the minimum intensity per RT range and force endpoints to zero.
    major_ticks : float or None, optional
        Spacing for major x-axis ticks.
    minor_ticks : float or None, optional
        Spacing for minor x-axis ticks.
    max_points : int or None, optional
        Maximum number of points per trace; if exceeded, downsample.

    Returns
    -------
    go.Figure or list of go.Figure or dict
        In overlay mode, one figure per dataset (or a single figure if one dataset);
        otherwise, separate figures per target m/z.
    """
    if isinstance(df, dict):
        dfs = []
        for key, value in df.items():
            value.name = key
            dfs.append(value)
    elif not isinstance(df, list):
        dfs = [df]
    else:
        dfs = df
    if not isinstance(target_mzs, (list, tuple, np.ndarray)):
        target_mzs = [target_mzs]
    if save_plots and not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    def downsample_data(df_subset, max_points):
        if max_points is not None and len(df_subset) > max_points:
            step = max(1, len(df_subset) // max_points)
            return df_subset.iloc[::step]
        return df_subset
    figures = []
    if overlay:
        for d in dfs:
            fig = go.Figure()
            chrom_df = extract_chromatograms(d, target_mzs, mz_tolerance)
            if rt_ranges is not None:
                rt_min, rt_max = rt_ranges
                chrom_df = chrom_df[(chrom_df['retention_time'] >= rt_min) &
                                    (chrom_df['retention_time'] <= rt_max)]
            dataset_name = getattr(d, 'name', "")
            trace_y_values = []
            for target, group in chrom_df.groupby('target_mz'):
                group = group.sort_values('retention_time')
                y_data = _apply_smoothing(group['intensity'], smooth_method, smooth_points)
                trace_y_values.append(np.array(y_data))
                trace_name = f"target m/z: {target}"
                fig.add_trace(go.Scattergl(
                    x=group['retention_time'],
                    y=y_data,
                    mode='lines',
                    name=trace_name,
                    hovertemplate='Retention Time: %{x}<br>Intensity: %{y}<extra></extra>'
                ))
            if trace_y_values:
                global_max = max(np.max(arr) for arr in trace_y_values if arr.size > 0)
            else:
                global_max = 1
            scale_factor = 1
            if global_max > 0:
                scale_factor = 10 ** int(np.floor(np.log10(global_max)))
            for i, y_vals in enumerate(trace_y_values):
                scaled = y_vals / scale_factor
                fig.data[i].y = scaled
            yaxis_title = "Intensity" if scale_factor == 1 else f"Intensity (×{scale_factor:0.0e})"
            title_text = f"Extracted Ion Chromatograms for {dataset_name}" if dataset_name else "Extracted Ion Chromatograms"
            fig.update_layout(
                title=title_text,
                xaxis_title="Retention Time",
                yaxis_title=yaxis_title,
                paper_bgcolor="white",
                plot_bgcolor="white",
                hovermode="closest",
                xaxis=dict(showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black')),
                yaxis=dict(showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black'), tickformat=".1f")
            )
            if major_ticks is not None:
                fig.update_xaxes(dtick=major_ticks)
            if minor_ticks is not None:
                fig.update_xaxes(minor=dict(dtick=minor_ticks, showgrid=True))
            if save_plots:
                save_path = os.path.join(save_directory, f"{dataset_name}_overlayed_mobilograms.html")
                fig.write_html(save_path)
            fig.show()
            figures.append(fig)
        return figures if len(figures) > 1 else figures[0]
    else:
        figs = {}
        color_sequence = px.colors.qualitative.Plotly
        unique_rt_ranges = sorted([f"{start}-{end}" for start, end in rt_ranges])
        rt_colors = {rt: color_sequence[i % len(color_sequence)] for i, rt in enumerate(unique_rt_ranges)}
        for d in dfs:
            dataset_name = getattr(d, 'name', "")
            chrom_df = extract_chromatograms(d, target_mzs, mz_tolerance)
            if rt_ranges is not None:
                rt_min, rt_max = rt_ranges
                chrom_df = chrom_df[(chrom_df['retention_time'] >= rt_min) &
                                    (chrom_df['retention_time'] <= rt_max)]
            for target in chrom_df['target_mz'].unique():
                sub_df = chrom_df[chrom_df['target_mz'] == target].sort_values('retention_time')
                fig = go.Figure()
                y_data = _apply_smoothing(sub_df['intensity'], smooth_method, smooth_points)
                y_arr = np.array(y_data)
                local_max = np.max(y_arr) if y_arr.size > 0 else 1
                scale_factor = 1
                if local_max > 0:
                    scale_factor = 10 ** int(np.floor(np.log10(local_max)))
                scaled_y = y_arr / scale_factor
                trace_name = f"target m/z: {target}" if dataset_name == "" else f"{dataset_name} target m/z: {target}"
                fig.add_trace(go.Scattergl(
                    x=sub_df['retention_time'],
                    y=scaled_y,
                    mode='lines',
                    name=trace_name,
                    hovertemplate='Retention Time: %{x}<br>Intensity: %{y}<extra></extra>'
                ))
                yaxis_title = "Intensity" if scale_factor == 1 else f"Intensity (×{scale_factor:0.0e})"
                title_text = f"Chromatogram for {dataset_name} - Target m/z: {target}" if dataset_name else f"Chromatogram for target m/z: {target}"
                fig.update_layout(
                    title=title_text,
                    xaxis_title="Retention Time",
                    yaxis_title=yaxis_title,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    hovermode="closest",
                    xaxis=dict(showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black')),
                    yaxis=dict(showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black'), tickformat=".1f")
                )
                if major_ticks is not None:
                    fig.update_xaxes(dtick=major_ticks)
                if minor_ticks is not None:
                    fig.update_xaxes(minor=dict(dtick=minor_ticks, showgrid=True))
                if save_plots:
                    save_path = os.path.join(save_directory, f"{dataset_name}_target_{target}_mobilogram.html")
                    fig.write_html(save_path)
                fig.show()
                figs[target] = fig
        return figs


def process_mobilogram_plots(lcms_data_dict, target_mzs, rt_ranges, normalize,
                             mz_tolerance=0.02, apply_smoothing=True, sigma=1,
                             overlay=False, save_plots=False, save_directory="plots",
                             baseline_correction=True):
    """
    Process LC–MS mobilogram plots for each sample in a dictionary.
    
    For each sample, mobilogram data is extracted and plotted using plot_mobilogram_target_mz.
    The function logs the sample being processed and returns a dictionary of figures.

    Parameters
    ----------
    lcms_data_dict : dict
        Dictionary of LC–MS DataFrames with sample names as keys.
    target_mzs : list of float
        Target m/z values.
    rt_ranges : list of tuple
        Retention time ranges.
    normalize : bool
        Whether to normalize intensities.
    mz_tolerance : float, optional
        m/z tolerance (default: 0.02).
    apply_smoothing : bool, optional
        If True, apply smoothing (default: True).
    sigma : float, optional
        Smoothing sigma (default: 1).
    overlay : bool, optional
        If True, overlay mobilograms (default: False).
    save_plots : bool, optional
        If True, save plots to disk (default: False).
    save_directory : str, optional
        Directory to save plots (default: "plots").
    baseline_correction : bool, optional
        Whether to apply baseline correction (default: True).

    Returns
    -------
    dict
        Dictionary of figures keyed by sample name.
    """
    import logging
    figures = {}
    for key, df in lcms_data_dict.items():
        logging.info(f"Plotting sample: {key}")
        fig = plot_mobilogram_target_mz(
            df=df,
            target_mzs=target_mzs,
            rt_ranges=rt_ranges,
            mz_tolerance=mz_tolerance,
            apply_smoothing=apply_smoothing,
            sigma=sigma,
            normalize=normalize,
            normalize_range=(0, 100),
            overlay=overlay,
            save_plots=save_plots,
            save_directory=save_directory,
            baseline_correction=baseline_correction
        )
        figures[key] = fig
    return figures


def process_mobilogram_interactive_plots(lcms_data_dict, target_mzs, rt_ranges, normalize,
                                         mz_tolerance=0.02, apply_smoothing=True, sigma=1,
                                         overlay=False, save_plots=False, save_directory="plots",
                                         baseline_correction=True):
    """
    Process interactive LC–MS mobilogram plots for each sample in a dictionary.
    
    For each sample, mobilogram data is extracted and interactive plots are created using plot_interactive_chromatograms.
    The function logs the sample being processed and returns a dictionary of interactive figures.

    Parameters
    ----------
    lcms_data_dict : dict
        Dictionary of LC–MS DataFrames with sample names as keys.
    target_mzs : list of float
        Target m/z values.
    rt_ranges : list of tuple
        Retention time ranges.
    normalize : bool
        Whether to normalize intensities.
    mz_tolerance : float, optional
        m/z tolerance (default: 0.02).
    apply_smoothing : bool, optional
        If True, apply smoothing (default: True).
    sigma : float, optional
        Smoothing sigma (default: 1).
    overlay : bool, optional
        If True, overlay mobilograms (default: False).
    save_plots : bool, optional
        If True, save plots as HTML files (default: False).
    save_directory : str, optional
        Directory to save plots (default: "plots").
    baseline_correction : bool, optional
        Whether to apply baseline correction (default: True).

    Returns
    -------
    dict
        Dictionary of interactive Plotly figures keyed by sample name.
    """
    import logging
    figures = {}
    for key, df in lcms_data_dict.items():
        logging.info(f"Plotting sample: {key}")
        fig = plot_interactive_chromatograms(
            df=df,
            target_mzs=target_mzs,
            rt_ranges=rt_ranges,
            mz_tolerance=mz_tolerance,
            apply_smoothing=apply_smoothing,
            sigma=sigma,
            normalize=normalize,
            normalize_range=(0, 100),
            overlay=overlay,
            save_plots=save_plots,
            save_directory=save_directory,
            baseline_correction=baseline_correction
        )
        figures[key] = fig
    return figures


def calculate_fragment_mz(peptide, charge_states=[1, 2, 3], fragments_to_save=None):
    """
    Calculate b and y ion series for a peptide with given charge states.
    Applies C-terminal amidation correction if the peptide ends with "(-0.98)".

    Parameters
    ----------
    peptide : str
        Peptide sequence. If amidated, it ends with "(-0.98)".
    charge_states : list, optional
        Charge states to consider (default: [1, 2, 3]).
    fragments_to_save : list, optional
        If provided, only include ion types in this list.

    Returns
    -------
    dict
        Dictionary with ion types as keys (e.g. "b1+", "y2+") and lists of m/z values as values.
    """
    amidated = peptide.endswith("(-0.98)")
    clean_peptide = peptide.replace("(-0.98)", "") if amidated else peptide
    n = len(clean_peptide)
    fragment_ions = {}
    for z in charge_states:
        b_series = [mass.calculate_mass(sequence=clean_peptide[:i], ion_type='b', charge=z, monoisotopic=True)
                    for i in range(1, n)]
        if amidated:
            y_series = [mass.calculate_mass(sequence=clean_peptide[i:], ion_type='y', charge=z, monoisotopic=True) - (0.98 / z)
                        for i in range(1, n)]
        else:
            y_series = [mass.calculate_mass(sequence=clean_peptide[i:], ion_type='y', charge=z, monoisotopic=True)
                        for i in range(1, n)]
        fragment_ions[f"b{z}+"] = b_series
        fragment_ions[f"y{z}+"] = y_series
    if fragments_to_save is not None:
        fragment_ions = {k: v for k, v in fragment_ions.items() if k in fragments_to_save}
    return fragment_ions


def calculate_precursor_mz(peptide, charge_states=[1, 2, 3]):
    """
    Calculate precursor m/z values for a peptide for given charge states.
    Accounts for C-terminal amidation if the peptide ends with "(-0.98)".

    Parameters
    ----------
    peptide : str
        Peptide sequence. If amidated, it ends with "(-0.98)".
    charge_states : list of int, optional
        Charge states for calculation (default: [1, 2, 3]).

    Returns
    -------
    dict
        Dictionary with keys "M{charge}+" and corresponding precursor m/z values.
    """
    amidated = peptide.endswith("(-0.98)")
    clean_peptide = peptide.replace("(-0.98)", "") if amidated else peptide
    neutral_mass = mass.calculate_mass(sequence=clean_peptide, ion_type='M', monoisotopic=True)
    if amidated:
        neutral_mass -= 0.98
    precursor_mzs = {}
    for z in charge_states:
        precursor_mz = mass.calculate_mass(sequence=clean_peptide, ion_type='M', charge=z, monoisotopic=True)
        if amidated:
            precursor_mz -= (0.98 / z)
        precursor_mzs[f"M{z}+"] = precursor_mz
    return precursor_mzs


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


# End of module
