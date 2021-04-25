#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------#
#                                                                             #
#   Define global parameters to be used through out the program               #
#                                                                             #
#-----------------------------------------------------------------------------#
colors = ["blue", "red", "firebrick", "magenta", "chocolate", "black", 
          "olive", "yellow", "green", "brown", "lime", "cyan"]


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create heatmap according to given parameters.         #
#                                                                             #
#*****************************************************************************#
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
    
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    
    # Turn spines off and create white grid.
    #ax.spines.set_visible(False)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im, cbar


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to add annotations to the already created heatmap.       #
#                                                                             #
#*****************************************************************************#
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                      textcolors=("black", "white"),
                      threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create graphs from data provided.                     #
#                                                                             #
#*****************************************************************************#
def plot_graphs(experiments):
    #pass
    #create_graph_samples_vs_reward(experiments)
    #create_graph_accuracy_vs_reward(experiments[:8], tag="alpha_100_beta_1")
    #create_graph_accuracy_vs_reward(experiments[8:16], tag="alpha_01_beta_1")
    #create_graph_accuracy_vs_reward(experiments[16:24], tag="alpha_100_beta_3")
    #create_graph_accuracy_vs_reward(experiments[24:32], tag="alpha_01_beta_3")
    #create_graph_accuracy_vs_reward(experiments[32:40], tag="alpha_100_beta_5")
    #create_graph_accuracy_vs_reward(experiments[40:48], tag="alpha_01_beta_5")
    #create_graph_accuracy_vs_reward(experiments[48], tag="heter_beta_1")
    #create_graph_accuracy_vs_reward(experiments[49], tag="heter_beta_3")
    #create_graph_accuracy_vs_reward(experiments[50], tag="heter_beta_5")
    #create_graph_confid_vs_reward(experiments[:15], tag="beta_1")
    #create_graph_confid_vs_reward(experiments[15:30], tag="beta_3")
    #create_graph_confid_vs_reward(experiments[30:45], tag="beta_5")
    create_heatmap(experiments[0:64], tag="mnist_alpha_100")
    create_heatmap(experiments[64:128], tag="mnist_alpha_01")
    create_heatmap(experiments[128:192], tag="cifar_alpha_100")
    create_heatmap(experiments[192:256], tag="cifar_alpha_01")


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create # of samples vs reward graph.                  #
#                                                                             #
#*****************************************************************************#
def create_graph_samples_vs_reward(experiments):
    # prepare data for the graph
    for i, exp in enumerate(experiments):
        beta = exp.hyperparameters["r_beta"]
        yAxis = np.transpose(np.array(exp.results["reward_0"])).reshape(-1)
        # create xticks to display on graph
        xticks = [i+1 for i in range(exp.hyperparameters["n_workers"])]
        labels = exp.hyperparameters["worker_data"]
        # prepare a plot
        fig, ax = plt.subplots()
        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")
        # plot actual graph        
        plt.xticks(xticks, labels)
        plt.plot(xticks, yAxis, "x", color="blue")
        plt.plot(xticks, yAxis)
        plt.title("Local Training data vs Reward graph")
        plt.xlabel("dataset size")
        plt.ylabel("reward")
        plt.savefig("samples_vs_reward_beta_{}.png".format(beta), dpi=600)
    

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create accuracy reward graph.                         #
#                                                                             #
#*****************************************************************************#
def create_graph_accuracy_vs_reward(experiments, tag=""):
    # Experiment legends
    exp_legends = ["Test 1 (Accuracy: 60%)",
                   "Test 2 (Accuracy: 65%)",
                   "Test 3 (Accuracy: 70%)",
                   "Test 4 (Accuracy: 75%)", 
                   "Test 5 (Accuracy: 80%)",
                   "Test 6 (Accuracy: 85%)",
                   "Test 7 (Accuracy: 90%)",
                   "Test 7 (Accuracy: 95%)"]
    # prepare data for the graph
    fig, ax = plt.subplots()
    # go trough all experiments and plot them
    for i, exp in enumerate(experiments):
        xAxis = [exp.results["tr_accuracy_0_worker_{}".format(i)][0] * 100.0 
                 for i in range(exp.hyperparameters["n_workers"])]
        yAxis = np.transpose(np.array(exp.results["reward_0"])).reshape(-1).tolist()
        # plot actual graph
        ax.plot(xAxis, yAxis, ".", color=colors[i], label=exp_legends[i])
    # create infomatics of the graph
    plt.title("Accuracy vs Reward Graph") # (Heterogeneous Group)")
    plt.xlabel("accuracy")
    plt.ylabel("reward")
    # create legend
    #ax.legend(loc="upper left", frameon=True)
    # save the figure
    plt.savefig("accuracy_vs_reward_{}.png".format(tag), dpi=600)
    

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create # of samples vs reward graph.                  #
#                                                                             #
#*****************************************************************************#
def create_graph_confid_vs_reward(experiments, tag=""):
    # Experiment legends
    exp_legends = ["Blind",
                   "Confidence: 30%", 
                   "Confidence: 50%", 
                   "Confidence: 70%", 
                   "Confidence: 90%"]
    # prepare data for the graph
    fig, ax = plt.subplots()

    # go trough all experiments and collect results
    xAxis = np.array([0, 2, 4])
    yAxis = []
    for i, exp in enumerate(experiments):
        yAxis += [np.transpose(np.array(exp.results["reward_0"])).reshape(-1)[0]]

    # reshape yAxis
    yAxis = np.array(yAxis).reshape((3, 5)).T
    bar_position = -0.50
    for i, data in enumerate(yAxis):
        plt.bar(xAxis+bar_position, data, color=colors[i], width = 0.25, label=exp_legends[i])
        bar_position += 0.25
        

    # create infomatics of the graph
    ax.set_title("Confidence vs Reward Graph")
    ax.set_xlabel("alpha")
    ax.set_ylabel("reward")
    
    # setup axis ticks
    plt.xticks(xAxis, ("0.1", "1.0", "100.0"))
    
    # create legend
    plt.legend()
    
    # save the figure
    plt.tight_layout()
    plt.savefig("confid_vs_reward_{}.png".format(tag), dpi=600)
        

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper functions to create heatmap graph from all experiments.            #
#                                                                             #
#*****************************************************************************#
def create_heatmap(experiments, tag=""):
    d1 = []
    xlabels = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800])
    ylabels = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800])
    
    # prepare data for the graph
    for exp in experiments:
        #n_distill = exp.hyperparameters["n_distill"]
        #epochs = exp.hyperparameters["local_epochs"]
        accuracy = exp.results["worker_0_accuracy"]
        d1.append(accuracy)
    
    data = np.array(d1).reshape([len(xlabels), len(ylabels)])
    print(data)
    print(data.shape)
    data = np.transpose(data)
    print(data)
    print(data.shape)
    
    # draw the heat map.
    fig, ax = plt.subplots()
    
    im = ax.imshow(data)
    
    im, cbar = heatmap(data, ylabels, xlabels, ax=ax, cmap="YlGn", cbarlabel="")
    annotate_heatmap(im, valfmt="{x:.2f}")
    
    fig.tight_layout()
    plt.savefig("heatmap_{}.png".format(tag), dpi=600)
    
    