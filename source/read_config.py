import yaml
import pathlib

class Config:

    # TODO (sameer): document this class

    # where the root directory is relative to this file
    root_dir = pathlib.Path("..")

    def __init__(self, yaml_filename):
        self.data =  self._load(yaml_filename)

    def _load(self, yaml_filename):
        with open(yaml_filename, 'r') as fh:
            yaml_data = yaml.safe_load(fh)
        return yaml_data

    def __str__(self):
        properties = [attr for attr in dir(self) 
                        if isinstance(getattr(Config,attr, None),property)]
        properties.sort()
        pstr_list = ["{0}: {1}".format(p, getattr(self, p)) for p in properties]
        return "\n".join(pstr_list)

    def safe_get_key(self, key, default=""):
        ret = default
        try:
            ret = self.data[key]
        except KeyError:
            print("Cannot get : {0}", key)
            pass
        return (ret)

    @property
    def aligned_msa_filename(self):
        return self.safe_get_key('aligned_msa_filename')

    @property
    def reweighting_threshold(self):
        return float(self.safe_get_key('reweighting_threshold', 0.8))

    @property
    def aligned_msa_fullpath(self):
        return self.dataset_dir / self.aligned_msa_filename

    @property
    def weights_fullpath(self):
        return (self.working_dir / pathlib.Path(self.aligned_msa_filename).stem
                    ).with_suffix('.npy')

    @property
    def dataset_dir(self):
        return Config.root_dir / pathlib.Path(self.safe_get_key('dataset_dir', 
                'sequence_sets'))

    @property
    def working_dir(self):
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


