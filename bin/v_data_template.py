import pandas as pd

class DataBaseLoader():
    def __init__(self):
        pass

    def load_files(self, key, *filenames):
        dfs = []
        for name in filenames:
            data = pd.read_csv(name)
            dfs.append(data)
        
        return self.load_data_frame(key, dfs)
    
    def load_data_frame(self, key, *dfs):
        output = dfs[0]
        for i in range(1, len(dfs)):
            output = pd.merge(output, dfs[i], on=key)
        return output