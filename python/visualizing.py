import plotly.graph_objects as go


def plot_collection(plot_only_this, distances_collection,
                    number_of_desired_plots=0):
    """
  Plot an iterable of tuples(label, time-series) until a desired number
  of plots number_of_desired_plots is reached. Plot all when 0.
  """
    fig = go.Figure()
    plots_count = 0
    print("total number of graphs: ", len(distances_collection))

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

    print("plots: ", plots_count)
    return fig
