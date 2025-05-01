import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from .lctims import *
pio.renderers.default = "browser"

def _set_axis_ticks(ax, major_ticks, minor_ticks):
    """
    Set the x-axis tick locators for the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure.
    major_ticks : int or None
        Spacing of major ticks.
    minor_ticks : int or None
        Spacing of minor ticks.
    """
    from matplotlib.ticker import MultipleLocator
    if major_ticks:
        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
    if minor_ticks:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

def _apply_smoothing(intensity_series, smooth_method, smooth_points):
    if smooth_method == 'savgol':
        # Ensure the window length is at least 3
        win_len = max(smooth_points, 3)
        # Ensure the window length is odd; if even, decrement by 1
        if win_len % 2 == 0:
            win_len -= 1

        n_points = len(intensity_series)
        if n_points < win_len:
            # If there are not enough points, adjust win_len to be the maximum odd number <= n_points
            adjusted_win_len = n_points if n_points % 2 == 1 else n_points - 1
            if adjusted_win_len < 3:
                # Not enough points to apply smoothing reliably; return original values
                return intensity_series.values
            else:
                win_len = adjusted_win_len

        try:
            return savgol_filter(intensity_series.values, window_length=win_len, polyorder=2)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
            return intensity_series.values
    else:
        return intensity_series.values



def plot_chromatograms(df, target_mzs, mz_tolerance=0.05, overlay=True, retention_time_range=None,
                       smooth_method=None, smooth_points=5, major_ticks=None, minor_ticks=None,
                       display=True, savefig_path=None):
    """
    Extract and plot chromatograms (EICs) for specified target m/z values with optional smoothing.
    Handles a single DataFrame, a list of DataFrames, or a dict of DataFrames.
    
    In overlay mode, one figure is created per dataset with all target m/z traces overlaid.

    Parameters
    ----------
    df : pd.DataFrame, list of pd.DataFrame, or dict
        DataFrame(s) with columns: 'frame', 'polarity', 'retention_time', 'mz', 'intensity', and 'mobility'.
        If a dict is provided, its keys will be used as dataset names.
    target_mzs : float or list of float
        Target m/z value(s).
    mz_tolerance : float, optional
        m/z matching tolerance (default: 0.05).
    overlay : bool, optional
        If True, overlay all target chromatograms in one figure per dataset; otherwise, create separate figures.
    retention_time_range : tuple or None, optional
        (rt_min, rt_max) to restrict plotted retention times.
    smooth_method : str or None, optional
        Smoothing method ('gaussian' or 'savgol'). Default: None.
    smooth_points : int, optional
        Number of points for smoothing (default: 5).
    major_ticks : int or None, optional
        Spacing for major x-axis ticks.
    minor_ticks : int or None, optional
        Spacing for minor x-axis ticks.
    display : bool, optional
        If True, display the figure (default: True).
    savefig_path : str or None, optional
        File path to save the figure. Supports format placeholders (default format: SVG).

    Returns
    -------
    matplotlib.figure.Figure or list of matplotlib.figure.Figure
        Figure(s) generated.
    """
    # Ensure df is a list of DataFrames.
    if isinstance(df, dict):
        dfs = []
        for key, value in df.items():
            value.name = key
            dfs.append(value)
    elif not isinstance(df, list):
        dfs = [df]
    else:
        dfs = df

    # Ensure target_mzs is a list.
    if not isinstance(target_mzs, (list, tuple, np.ndarray)):
        target_mzs = [target_mzs]

    if savefig_path is not None and not os.path.exists(os.path.dirname(savefig_path)):
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)

    figures = []
    if overlay:
        for d in dfs:
            fig, ax = plt.subplots(figsize=(10, 6))
            chrom_df = extract_chromatograms(d, target_mzs, mz_tolerance)
            if retention_time_range is not None:
                rt_min, rt_max = retention_time_range
                chrom_df = chrom_df[(chrom_df['retention_time'] >= rt_min) & (chrom_df['retention_time'] <= rt_max)]
            dataset_name = getattr(d, 'name', "")
            for target, group in chrom_df.groupby('target_mz'):
                group = group.sort_values('retention_time')
                smoothed_intensity = _apply_smoothing(group['intensity'], smooth_method, smooth_points)
                ax.plot(group['retention_time'], smoothed_intensity, linestyle='-', label=f"target m/z: {target}")
            _set_axis_ticks(ax, major_ticks, minor_ticks)
            ax.set_xlabel("Retention Time")
            ax.set_ylabel("Intensity")
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            title = f"Extracted Ion Chromatograms for {dataset_name}" if dataset_name else "Extracted Ion Chromatograms"
            ax.set_title(title)
            ax.legend()
            if savefig_path is not None:
                plt.rcParams['svg.fonttype'] = 'none'
                fig.savefig(savefig_path.format(dataset=dataset_name), format='svg')
            if display:
                plt.show()
            figures.append(fig)
        return figures if len(figures) > 1 else figures[0]
    else:
        figs = []
        for d in dfs:
            chrom_df = extract_chromatograms(d, target_mzs, mz_tolerance)
            if retention_time_range is not None:
                rt_min, rt_max = retention_time_range
                chrom_df = chrom_df[(chrom_df['retention_time'] >= rt_min) & (chrom_df['retention_time'] <= rt_max)]
            dataset_name = getattr(d, 'name', "")
            for target in chrom_df['target_mz'].unique():
                sub_df = chrom_df[chrom_df['target_mz'] == target].sort_values('retention_time')
                fig, ax = plt.subplots(figsize=(10, 6))
                smoothed_intensity = _apply_smoothing(sub_df['intensity'], smooth_method, smooth_points)
                ax.plot(sub_df['retention_time'], smoothed_intensity, linestyle='-')
                _set_axis_ticks(ax, major_ticks, minor_ticks)
                ax.set_xlabel("Retention Time")
                ax.set_ylabel("Intensity")
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                title = f"{dataset_name} Chromatogram for Target m/z: {target}" if dataset_name else f"Chromatogram for Target m/z: {target}"
                ax.set_title(title)
                ax.legend()
                if savefig_path is not None:
                    plt.rcParams['svg.fonttype'] = 'none'
                    fig.savefig(savefig_path.format(dataset=dataset_name, target=target), format='svg')
                if display:
                    plt.show()
                figs.append(fig)
        return figs


def plot_interactive_chromatograms(df, target_mzs, mz_tolerance=0.05, overlay=True, 
                                   retention_time_range=None, smooth_method=None, smooth_points=5):
    """
    Extracts and plots interactive chromatograms (EICs) for specified target m/z values using Plotly.
    
    The function creates interactive line plots that allow zooming, panning, and hovering.
    It uses the actual DataFrame name (via the 'name' attribute) in the title and legend.
    Additionally, it scales the y-values so that the ticks are displayed in a more readable format.
    For example, if the maximum intensity is 1e6, all intensities are divided by 1e6 so that 500000 is
    shown as 0.5, 1000000 as 1.0, etc.
    
    Parameters
    ----------
    df : pd.DataFrame, list of pd.DataFrame, or dict
        Input DataFrame(s) containing columns: 'frame', 'polarity', 'retention_time', 'mz', 
        'intensity', and 'mobility'. If a dict is provided, its keys will be used as the dataset names.
    target_mzs : float or list of float
        A single target m/z or a list of target m/z values.
    mz_tolerance : float, optional
        Tolerance for m/z matching (default is 0.05).
    overlay : bool, optional
        If True, for each dataset, overlay all target chromatograms in a single figure;
        if False, create separate figures for each target m/z.
    retention_time_range : tuple or None, optional
        A tuple (rt_min, rt_max) to restrict the plotted retention times.
    smooth_method : str or None, optional
        Smoothing method ('gaussian' or 'savgol'). If None, no smoothing is applied.
    smooth_points : int, optional
        Number of points (or sigma value) for smoothing (default is 5).
    
    Returns
    -------
    plotly.graph_objects.Figure or list of plotly.graph_objects.Figure or dict
        In overlay mode, a single figure is returned if only one dataset is provided,
        otherwise a list of figures (one per dataset). In non-overlay mode, a dictionary of figures
        keyed by target m/z is returned.
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # If df is a dict, convert it to a list of DataFrames and set each DataFrame's name.
    if isinstance(df, dict):
        dfs = []
        for key, value in df.items():
            value.name = key
            dfs.append(value)
    elif not isinstance(df, list):
        dfs = [df]
    else:
        dfs = df

    if overlay:
        figures = []
        # Create one figure per dataset.
        for d in dfs:
            fig = go.Figure()
            chrom_df = extract_chromatograms(d, target_mzs, mz_tolerance)
            if retention_time_range is not None:
                rt_min, rt_max = retention_time_range
                chrom_df = chrom_df[(chrom_df['retention_time'] >= rt_min) &
                                    (chrom_df['retention_time'] <= rt_max)]
            dataset_name = getattr(d, 'name', "")
            trace_y_values = []
            # Group by target m/z (each trace in this figure corresponds to a target m/z).
            for target, group in chrom_df.groupby('target_mz'):
                group = group.sort_values('retention_time')
                y_data = _apply_smoothing(group['intensity'], smooth_method, smooth_points)
                trace_y_values.append(np.array(y_data))
                # Since this figure is for one dataset, we list only the target m/z in the legend.
                trace_name = f"target m/z: {target}"
                fig.add_trace(go.Scatter(
                    x=group['retention_time'],
                    y=y_data,  # will be scaled later
                    mode='lines',
                    name=trace_name,
                    hovertemplate='Retention Time: %{x}<br>Intensity: %{y}<extra></extra>'
                ))
            # Compute the global maximum for this dataset.
            global_max = max(np.max(arr) for arr in trace_y_values if arr.size > 0)
            scale_factor = 1
            if global_max > 0:
                scale_factor = 10 ** int(np.floor(np.log10(global_max)))
            # Update each trace's y values with the scaled data.
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
                xaxis=dict(
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(color='black'),
                    tickformat=".1f"
                )
            )
            figures.append(fig)
            fig.show()
        return figures 

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

def _apply_smoothing(intensity_series, smooth_method, smooth_points):
    if smooth_method == 'savgol':
        # Ensure the window length is at least 3
        win_len = max(smooth_points, 3)
        # Ensure the window length is odd; if even, decrement by 1
        if win_len % 2 == 0:
            win_len -= 1

        n_points = len(intensity_series)
        if n_points < win_len:
            # If there are not enough points, adjust win_len to be the maximum odd number <= n_points
            adjusted_win_len = n_points if n_points % 2 == 1 else n_points - 1
            if adjusted_win_len < 3:
                # Not enough points to apply smoothing reliably; return original values
                return intensity_series.values
            else:
                win_len = adjusted_win_len

        try:
            return savgol_filter(intensity_series.values, window_length=win_len, polyorder=2)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
            return intensity_series.values
    else:
        return intensity_series.values



def plot_mobilogram_target_mz(df, target_mzs, rt_ranges, mz_tolerance=0.1, 
                              apply_smoothing=False, sigma=1, 
                              normalize=True, normalize_range=(0, 1),
                              overlay=False, save_plots=False, save_directory="plots",
                              baseline_correction=True):
    """
    Extract mobilogram data for target m/z values using extract_mobilogram and plot mobility vs. intensity.
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
            mob_df = extract_mobilogram(
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
            mob_df = extract_mobilogram(
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

def plot_mobilogram_interactive(df, target_mzs, rt_ranges, mz_tolerance=0.05, 
                                apply_smoothing=False, sigma=1, 
                                normalize=True, normalize_range=(0, 1),
                                overlay=False, save_plots=False, save_directory="plots",
                                baseline_correction=True,
                                major_ticks=None, minor_ticks=None,
                                max_points=None):
    """
    Extract mobilogram data for one or more target m/z values using the extract_mobilogram function,
    and create interactive Plotly plots of mobility versus intensity using WebGL (Scattergl) for performance.
   
    Parameters
    ----------
    df : pd.DataFrame or list of pd.DataFrame
        Input DataFrame(s) containing columns: 'retention_time', 'mz', 'intensity', and 'mobility'.
    target_mzs : float or list of float
        A target m/z value or list of target m/z values.
    rt_ranges : list of tuple
        List of (start_rt, end_rt) tuples used in extract_mobilogram.
    mz_tolerance : float, optional
        Tolerance for m/z matching (default is 0.1).
    apply_smoothing : bool, optional
        If True, apply Gaussian smoothing during mobilogram extraction (default is False).
    sigma : float, optional
        Standard deviation for Gaussian smoothing (default is 1).
    normalize : bool, optional
        If True, normalize intensities within each RT range to the specified normalize_range.
    normalize_range : tuple, optional
        Range to which intensities are normalized (default is (0, 1)).
    overlay : bool, optional
        If True, plot all target mobilograms in a single interactive figure; otherwise, create separate figures.
    save_plots : bool, optional
        If True, save each figure as an HTML file.
    save_directory : str, optional
        Directory in which to save plots if save_plots is True.
    baseline_correction : bool, optional
        If True, subtract the minimum intensity from each RT range to set the baseline to zero.
        In addition, the first and last intensity values in each RT range are forced to zero.
    major_ticks : float or None, optional
        Spacing for major ticks on the x-axis. If None, default Plotly settings are used.
    minor_ticks : float or None, optional
        Spacing for minor ticks on the x-axis. If None, default Plotly settings are used.
    max_points : int or None, optional
        Maximum number of points to plot per trace. If the number of data points exceeds max_points,
        the data will be downsampled by taking every nth point.
        
    Returns
    -------
    list of plotly.graph_objects.Figure
        List of generated interactive Plotly figures.
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    # Determine the dataset name(s) from the input DataFrame(s)
    if isinstance(df, list):
        dataset_names = [getattr(d, 'name', f"Dataset {i+1}") for i, d in enumerate(df)]
        combined_dataset_name = ", ".join(dataset_names)
    else:
        combined_dataset_name = getattr(df, 'name', "Dataset 1")
    
    # Ensure target_mzs is a list.
    if not isinstance(target_mzs, (list, tuple, np.ndarray)):
        target_mzs = [target_mzs]
    
    # Create the save directory if needed.
    if save_plots and not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    def downsample_data(df_subset, max_points):
        """Downsample the dataframe if it has more than max_points rows."""
        if max_points is not None and len(df_subset) > max_points:
            step = max(1, len(df_subset) // max_points)
            return df_subset.iloc[::step]
        return df_subset

    figures = []
    
    if overlay:
        # Create one interactive Plotly figure with a white background.
        fig = go.Figure()
        # Create a color mapping for each target m/z.
        color_sequence = px.colors.qualitative.Plotly
        n_targets = len(target_mzs)
        target_colors = {target: color_sequence[i % len(color_sequence)] for i, target in enumerate(target_mzs)}
        
        # Loop over each target m/z.
        for target in target_mzs:
            # Extract mobilogram data.
            mob_df = extract_mobilogram(
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
            # Loop over each RT range for this target.
            for rt in sorted(mob_df['rt_range'].unique()):
                subset = mob_df[mob_df['rt_range'] == rt].sort_values('mobility')
                if not subset.empty:
                    # Force first and last intensity values to zero.
                    subset.loc[subset.index[0], 'normalized_intensity'] = 0
                    subset.loc[subset.index[-1], 'normalized_intensity'] = 0
                    # Downsample if necessary.
                    subset = downsample_data(subset, max_points)
                    fig.add_trace(
                        go.Scattergl(
                            x=subset['mobility'],
                            y=subset['normalized_intensity'],
                            mode='lines',
                            name=f"m/z {target}, RT {rt}",
                            line=dict(color=target_colors[target])
                        )
                    )
        # Update layout with visible axis lines and tick labels.
        fig.update_layout(
            title="Overlayed Mobilograms for " + combined_dataset_name + (" (Normalized)" if normalize else ""),
            xaxis_title="Mobility",
            yaxis_title="Normalized Intensity" if normalize else "Intensity",
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend_title="Target m/z and RT Range",
            margin=dict(r=150),
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                tickfont=dict(color='black')
            )
        )
        if major_ticks is not None:
            fig.update_xaxes(dtick=major_ticks)
        if minor_ticks is not None:
            fig.update_xaxes(minor=dict(dtick=minor_ticks, showgrid=True))
            
        if save_plots:
            filename = os.path.join(save_directory, "overlayed_mobilograms.html")
            fig.write_html(filename)
        fig.show()
        figures.append(fig)
        
    else:
        # Create separate interactive figures for each target m/z.
        color_sequence = px.colors.qualitative.Plotly
        unique_rt_ranges = sorted([f"{start}-{end}" for start, end in rt_ranges])
        rt_colors = {rt: color_sequence[i % len(color_sequence)] for i, rt in enumerate(unique_rt_ranges)}
        
        for target in target_mzs:
            fig = go.Figure()
            mob_df = extract_mobilogram(
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
                    subset = downsample_data(subset, max_points)
                    fig.add_trace(
                        go.Scattergl(
                            x=subset['mobility'],
                            y=subset['normalized_intensity'],
                            mode='lines',
                            name=f"RT {rt}",
                            line=dict(color=rt_colors[f"{rt}"])
                        )
                    )
            fig.update_layout(
                title=f"Mobilogram for {combined_dataset_name} - Target m/z: {target}" + (" (Normalized)" if normalize else ""),
                xaxis_title="Mobility",
                yaxis_title="Normalized Intensity" if normalize else "Intensity",
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend_title="RT Range",
                margin=dict(r=150),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    tickfont=dict(color='black')
                )
            )
            if major_ticks is not None:
                fig.update_xaxes(dtick=major_ticks)
            if minor_ticks is not None:
                fig.update_xaxes(minor=dict(dtick=minor_ticks, showgrid=True))
                
            if save_plots:
                filename = os.path.join(save_directory, f"target_{target}_mobilogram.html")
                fig.write_html(filename)
            fig.show()
            figures.append(fig)
    
    return figures


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
        fig = plot_mobilogram_interactive(
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
