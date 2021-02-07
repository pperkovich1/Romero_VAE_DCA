import resource
import matplotlib.pyplot as plt

def plot_loss_curve(losses, annotatation_str="", save_fig_path=None,
        model_name="", ax=None, save_fig_kws={"dpi":300}, xlabel="Epoch",
        ylabel="loss", title="", bbox= {"boxstyle":"round", "fc":"0.8"}):
    """ Save graph of loss curves """
    if ax is None:
        ax = plt.figure().gca()
    ax.plot(losses, "o-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not title: # (title is empty)
        title = f"Loss Curve :{model_name}"
    ax.set_title(title)
    if annotatation_str != "":
        ax.annotate(annotatation_str, (0.5, 0.5), xycoords='axes fraction',
                bbox=bbox);
    if save_fig_path is not None:
        plt.savefig(save_fig_path, **save_fig_kws)

def get_input_length(dataset):
    sample = dataset.__getitem__(0)[0]
    return len(sample)

def sizeof_fmt(num, suffix='B'):
    """ Copied from 
    https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_max_memory_usage(human_string=True):
    """ Returns memory usage in kilobytes"""
    ret = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if human_string:
        ret = sizeof_fmt(ret * 1024)
    return ret
