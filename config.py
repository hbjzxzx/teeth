import yaml
import os
class config(object):
    def __init__(self, path=None, dc=None):
        self.configDic = {}
        if path:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.configDic = yaml.load(f, Loader=yaml.FullLoader)
            else:
                print('construct config object failed,  check the config file path')
        if dc:
            self.mergeDict(self.configDic, dc)
          

    def mergeDict(self, dcOri:dict, dcNew:dict):
        for key, NewVal in dcNew.items():
            if key in dcOri.keys():
                OriVal = dcOri[key]
                if isinstance(OriVal, dict) and isinstance(NewVal, dict):
                    # two dict merge
                    self.mergeDict(OriVal, NewVal)
                else:
                    dcOri[key] = NewVal
                    print(f'warning: {key} value has been replace by {NewVal}, original is {OriVal}')
            else:
                # add new Key
                dcOri[key] = NewVal 


    def mergeWith(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                d = yaml.load(f, Loader=yaml.FullLoader)
        self.mergeDict(self.configDic, d)


    def __getitem__(self, item):
        value = self.configDic[item]

        if isinstance(value, dict):
            subConfig = config(dc=value)
            spItem='rangeMode'
            if spItem in value.keys():
                subConfig.rangeMode = value[spItem]
            else:
                subConfig.rangeMode = self.rangeMode
            return subConfig

        else:
            # return value
            if isinstance(value, list):
                # 'totalRange and tseedRange always use interval range'
                if item == 'totalRange' or item == 'tseedRange':
                    return list(range(*value))

                if self.rangeMode == 'interval':
                    return list(range(*value))
                elif self.rangeMode == 'list':
                    return value 
                else:
                    raise Exception(f'error rangeMode setting: {self.rangeMode} must be "interval" or "list"')
            return value
              
    def __str__(self):
        return str(self.configDic)
    
    def __repr__(self):
        return self.configDic.__repr__() 

