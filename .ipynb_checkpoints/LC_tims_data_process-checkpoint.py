#!/usr/bin/env python
"""
Module: timstof_analysis
Description: Functions for analyzing TIMSTOF data.
"""

import os
import re
from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
from pyTDFSDK import *
from pyTDFSDK.classes import TdfData, CHROMATOGRAM_JOB_GENERATOR
from pyTDFSDK.tsf import tsf_read_line_spectrum_v2, tsf_index_to_mz
from pyTDFSDK.ctypes_data_structures import PressureCompensationStrategy
from pyteomics import mass


#initialize library
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


