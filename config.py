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

    def mergeWith(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                d = yaml.load(f, Loader=yaml.FullLoader)
            for key, val in d.items():
                if key in self.configDic.keys():
                    print(f"warning replace the setting {key} from {self.configDic[key]} to {d[key]}")
                self.configDic[key] = d[key]

    def __getitem__(self, item):
        value = self.configDic[item]
        # transform string to number
        if isinstance(value, str) and value.isdigit():
            try:
                x = int(value)
            except ValueError:
                x = float(value)
            finally:
                return x
        elif isinstance(value, str):
            value = value.lower()
            # handle specific string

            #handle boolean
            if value == 'on' or value == 'true':
                return True
            elif value == 'off' or value == 'false':
                return False
        return value


