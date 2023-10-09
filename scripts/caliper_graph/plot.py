from matplotlib import rcParams
import os

rcParams.update({'figure.autolayout': True})

KINDS = ["bar", "hist", "line"]
METRICS = {"bandwidth": "bandwidth(GiB/s)",
           "FLOPS": "GFLOPS",
           "speedup": "time(s)/time(s)",
           "throughput": "throughput(GProblemsize/s)",
           "time/rep": "time/rep(s)",
}


def _derive_bandwidth(thicket, header):
    return thicket.dataframe[header, "Bytes/Rep"] / _derive_time(thicket, header) / 10**9


def _derive_flops(thicket, header):
    return thicket.dataframe[header, "Flops/Rep"] / _derive_time(thicket, header) / 10**9


def _derive_speedup(thicket, header_list):
    return thicket.dataframe[header_list[0], "Total time"] / thicket.dataframe[header_list[1], "Total time"]

def _derive_throughput(thicket, header):
    return thicket.dataframe[header, "ProblemSize"] / _derive_time(thicket, header) / 10**9


def _derive_time(thicket, header):
    return thicket.dataframe[header, "Total time"] / thicket.dataframe[header, "Reps"]


def _graph_bar(df, metric, prefix):
    num_xticks = len(df)
    plt = df.plot(kind="bar", 
                    title=f"{METRICS[metric]}", 
                    ylabel=METRICS[metric],
                    grid=True,
                    figsize=(max(num_xticks*0.5, 4), 6,),
        )
    plt.figure.savefig(f"{prefix}/bar_{metric}.png")


def _graph_hist(df, metric, prefix):
    num_xticks = len(df)
    plt = df.plot(kind="hist", 
                    title=f"{METRICS[metric]}",
                    xlabel=METRICS[metric],
                    grid=True,
                    figsize=(max(num_xticks*0.5, 4), 6,),
                    subplots=True,
        )
    plt[0].figure.savefig(f"{prefix}/hist_{metric}.png")


def _graph_line(df, metric, prefix, name):
    plt = df.plot(kind="line", 
                    marker='o', 
                    title=f"{name}", 
                    ylabel=METRICS[metric], 
                    logx=True,
                    logy=True,
                    grid=True,
        )
    plt.figure.savefig(f"{prefix}/{name}.png")


def plot(thicket, kind=None, metric=None, prefix=None):
    """Prepares dataframe for plotting and calls appropriate plotting function
    
    Arguments:
        thicket (Thicket): Thicket object
        kind (str): Type of plot to make
        metric (str): Metric to plot
        prefix (str): Prefix for output file
    
    Returns:
        df (DataFrame): Dataframe used for plotting
    """
    if kind is None:
        raise ValueError(f"kind must be specified from: {KINDS}")
    if metric is None:
        raise ValueError(f"metric must be specified from: {list(METRICS.keys())}")

    func = None
    if metric == "bandwidth":
        func = _derive_bandwidth
        if prefix is None:
            prefix = "graphs/graph_bandwidth"
    elif metric == "FLOPS":
        func = _derive_flops
        if prefix is None:
            prefix = "graphs/graph_flops"
    elif metric == "speedup":
        func = _derive_speedup
        if prefix is None:
            prefix = "graphs"
    elif metric == "throughput":
        func = _derive_throughput
        if prefix is None:
            prefix = "graphs/graph_throughput"
    elif metric == "time/rep":
        func = _derive_time
        if prefix is None:
            prefix = "graphs/graph_time"

    g_func = None
    if kind == "bar":
        g_func = _graph_bar
    elif kind == "hist":
        g_func = _graph_hist
    elif kind == "line":
        g_func = _graph_line

    # Make dir
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # Add calculated column to dataframe
    header_list = [h for h in thicket.dataframe.columns.get_level_values(0).unique() if "name" not in h]
    if metric == "speedup":
        thicket.dataframe[f"{header_list[0]}/{header_list[1]}", metric] = func(thicket, header_list)
    else:
        for header in header_list:
            thicket.dataframe[header, metric] = func(thicket, header)

    # Make copy
    df = thicket.dataframe.copy(deep=True)
    if kind == "bar" or kind == "hist":
        df.reset_index(inplace=True)
        drop_cols = [col for col in df.columns if not "name" in col and not metric in col]
        df.drop(columns=drop_cols, inplace=True)
        df.set_index([("name", "")], inplace=True)
        df.columns = df.columns.droplevel(1)
        g_func(df, metric, prefix)
    elif kind == "line":
        # Plot for each node
        for node in set(thicket.dataframe.index.get_level_values("node")):
            df = thicket.dataframe.loc[node]
            name = df[("name", "")].iloc[0]
            drop_cols = [col for col in df.columns if col[1] != metric or df[col].isnull().values.all()]
            df = df.drop(columns=drop_cols, axis=1)
            df.columns = df.columns.droplevel(1)
            g_func(df, metric, prefix, name)
    
    return df

