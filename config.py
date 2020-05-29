import yaml
import os
class config(object):
    def __init__(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.configDic = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print('construct config object failed, check the config file path')
            raise FileExistsError("file {} not exist".format(path))

    def __getitem__(self, item):
        value = self.configDic[item]
        if isinstance(value, str) and value.isdigit():
            try:
                x = int(value)
            except ValueError:
                x = float(value)
            finally:
                return x
        return value


