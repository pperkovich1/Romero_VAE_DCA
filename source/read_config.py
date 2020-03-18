import yaml
import pathlib

class Config:
    """Miscellaneous properties for config, derived from a config.yaml file

    To use:
    >>> config = Config(config_yaml_filename)
    >>> config.aligned_msa_fullpath
    PosixPath('../sequence_sets/cmx_aligned_blank_90.fasta')
    >>> config.reweighting_threshold
    0.8
    >>> config.weights_fullpath # combining properties
    PosixPath('../working/cmx_aligned_blank_90.npy')
    """

    # where the root directory is relative to this file
    root_dir = pathlib.Path("..")

    def __init__(self, yaml_filename):
        self.data =  self._load_yaml(yaml_filename)

    def _load_yaml(self, yaml_filename):
        """ Simple function to load yaml file """
        with open(yaml_filename, 'r') as fh:
            yaml_data = yaml.safe_load(fh)
        return yaml_data

    def __str__(self):
        """Search for properties of this class and return a string consisting
        of a sorted list of properties and values""" 
        properties = [attr for attr in dir(self) 
                        if isinstance(getattr(Config,attr, None),property)]
        properties.sort()
        pstr_list = ["{0}: {1}".format(p, getattr(self, p)) for p in properties]
        return "\n".join(pstr_list)

    def safe_get_key(self, key, default=""):
        """Get a key from the yaml file but if it doesn't exist then set it to default"""
        ret = default
        try:
            ret = self.data[key]
        except KeyError:
            print("Cannot get : {0}, setting default to : {1}", key, default)
            pass
        return (ret)

    @property
    def aligned_msa_filename(self):
        """String input msa filename from yaml file"""
        return self.safe_get_key('aligned_msa_filename')

    @property
    def reweighting_threshold(self):
        """Float re-weighting threshold (similarity to a sequence)"""
        return float(self.safe_get_key('reweighting_threshold', 0.8))

    @property
    def aligned_msa_fullpath(self):
        """Complete path to the input MSA"""
        return self.dataset_dir / self.aligned_msa_filename

    @property
    def weights_fullpath(self):
        """Complete path to the output weights npy file"""
        return (self.working_dir / pathlib.Path(self.aligned_msa_filename).stem
                    ).with_suffix('.npy')

    @property
    def dataset_dir(self):
        """Complete path to the dataset directory """
        return Config.root_dir / pathlib.Path(self.safe_get_key('dataset_dir', 
                'sequence_sets'))

    @property
    def working_dir(self):
        """ Complete path to the working directory """
        return Config.root_dir / pathlib.Path(self.safe_get_key('working_dir', 
                'sequence_sets'))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process yaml config file')
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)
    print(config)


