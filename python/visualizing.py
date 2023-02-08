import logging
import plotly.graph_objects as go
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def plot_collection(plot_only_this, distances_collection,
                    number_of_desired_plots=0):
    """
    Plot an iterable of tuples(label, time-series) until a desired number
    of plots number_of_desired_plots is reached. Plot all when 0.
    """
    fig = go.Figure()
    plots_count = 0
    logging.info(f"total number of graphs: {len(distances_collection)}")

    for (pattern_name, d) in distances_collection:
        if pattern_name == plot_only_this:
            if plots_count == 0:
                fig.add_trace(go.Line(y=d))
                fig.update_layout(title=plot_only_this)
            else:
                fig.add_trace(go.Line(y=d))
            plots_count += 1
            fig.update_layout(width=1200, height=800)
            if number_of_desired_plots:
                if plots_count == number_of_desired_plots:
                    break

    logging.info(f"plots: {plots_count}")
    return fig


def plot_outliers_in(df_single, y_label_name: str, outlier_name: str = 'outlier', column_name: str = 'reading'):
    """
    Plot the outliers in a dataframe with columns difined in the parameters.
    Use outlier_name for the column with the binary indication of outlier mathching
    and column name for the column that contain the time-series to plot. Use y_labe_name
    to deffin the name to show in the y-label of the plot.
    """
    a = df_single[df_single[outlier_name] == 1]
    fig = plt.figure(figsize=(15, 8))
    _ = plt.plot(df_single[column_name], color='blue', label='Normal')
    _ = plt.plot(a[column_name], linestyle='none', marker='X', color='red', markersize=12, label='Outlier')
    _ = plt.xlabel('Row Index (Experiment)')
    _ = plt.ylabel(f'{y_label_name} single experiment')
    _ = plt.title('EXTREME Experimental Outliers for Number of Data Samples per Experiment')
    _ = plt.legend(loc='best')
    return fig
