import numpy as np
import pandas as pd
import collections

class Continuous:
    
    def __init__(self,fileCSV=None, headerLabels=None):
        
        if fileCSV is None:
            self.pathBank = './dataset/dataset.csv'
            self.fileCSV = pd.read_csv(filepath_or_buffer=self.pathBank,delimiter = ',', header=0, index_col=0);
        else:
            self.fileCSV = pd.read_csv(filepath_or_buffer=fileCSV, delimiter = ';', header=0, names= headerLabels);
       
        self.continuous = self.fileCSV.select_dtypes(include=[np.number]);
        self.pathFeatures = './results/continuous-features.csv';
        self.pathDQR = './results/E-DQR-continuous.csv';    
    

    
    def write_results(self, table=None, header_columns=None, path=None):
        if table is None and path is None:
            pd.DataFrame(self.continuous).to_csv(path_or_buf=self.pathFeatures);
        else:
            pd.DataFrame(table).to_csv(path_or_buf= path, header= header_columns, index= False);
   
    def reduceAge(self,dataFeature):
        len(dataFeature)
        keyTabs = []
        for key,value in enumerate(self.fileCSV['age']):
            if value <= 16 or value >= 65:
                keyTabs.append(key)
        dataFeature = dataFeature.drop(dataFeature.index[keyTabs])
        return dataFeature
    
    def draw_DQR(self):
        
        self.__all_continous_header = [];
        self.__continuous_features_table = [];
        self.__continuous_header = ["Feature name","Count","% Miss", "Card", "Min", "1st Qrt", "Mean","Median","3rd Qrt","Max", "Standard deviation" ]
        continuous_columns = self.continuous;
        self.__all_continuous_table = collections.OrderedDict();
        tabExcludeCat = ['detailed occupation recode','detailed industry recode','own business or self employed','veteran\'s benefits'];
        for col, cat in enumerate(continuous_columns.keys()):
            hasString = False;
            countWrongItem = 0;
            dataFeature = self.fileCSV[cat];
            if cat in "age":
                dataFeature = self.reduceAge(dataFeature)
                
            for value in dataFeature:
                if type(value) is not int:
                    hasString = True;
                elif value == 0 and cat not in tabExcludeCat:
                    countWrongItem = countWrongItem+1;
            
            if hasString is False:
                
#                if  (np.max(dataFeature) - np.percentile(dataFeature, 75) >  np.percentile(dataFeature, 75) - np.percentile(dataFeature, 50)) or
#                    (np.percentile(dataFeature, 25) - np.min(dataFeature) >  np.percentile(dataFeature, 25) - np.percentile(dataFeature, 50)):
#                   
                if countWrongItem/ dataFeature.size * 100 < 60:
                    feature = collections.OrderedDict();
                    feature['nameFeature'] = cat;
                    feature['countTotal'] = dataFeature.size;                  
                    feature['% Miss'] = countWrongItem / dataFeature.size * 100;
                    feature['cardTotal'] = np.unique(dataFeature).size;
                    feature['min'] = np.min(dataFeature);
                    feature['firstQuarter'] = np.percentile(dataFeature, 25);
                    feature['mean'] = round(np.mean(dataFeature), 2);
                    feature['median'] = np.percentile(dataFeature, 50);
                    feature['thirdQuarter'] = np.percentile(dataFeature, 75);
                    feature['max'] = np.max(dataFeature);
                    feature['std'] = np.std(dataFeature);
                    self.__continuous_features_table.append(feature);
                    
                    self.__all_continous_header.append(cat);
                    self.__all_continuous_table[cat] = continuous_columns.values[:,col];
              
#        Write DQR continuous
        self.write_results(self.__continuous_features_table, self.__continuous_header, self.pathDQR);
        self.write_results(self.__all_continuous_table, self.__all_continous_header,"./results/continuous-features.csv");
