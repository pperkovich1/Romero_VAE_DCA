import resource

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
