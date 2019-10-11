import numpy as np
import pandas as pd
import collections

class Categorical:
    
    def __init__(self,fileCSV=None, headerLabels=None):
        
        if fileCSV is None:
            self.pathBank = './dataset/dataset.csv'
            self.fileCSV = pd.read_csv(filepath_or_buffer=self.pathBank,delimiter = ',', header=0, index_col=0);
        else:
            self.fileCSV = pd.read_csv(filepath_or_buffer=fileCSV, delimiter = ';', header=0, names= headerLabels);
       
        self.categorical = self.fileCSV.select_dtypes(exclude=[np.number]);
        self.pathFeatures = './results/-features.csv';
        self.pathDQR = './results/E-DQR-categorical.csv';    
    

    
    def write_results(self, table=None, header_columns=None, path=None):
        if table is None and path is None:
            pd.DataFrame(self.categorical).to_csv(path_or_buf=self.pathFeatures);
        else:
            pd.DataFrame(table).to_csv(path_or_buf= path, header= header_columns, index= False);
    


    
    def draw_DQR(self):

        self.__all_categorical_header = [];
        self.__continuous_features_table = [];
        self.__categorical_header = ["Feature name","Count","% Miss", "Card", "Mode", "Mode Freq", "Mode %","2nd Mode","2nd Mode Freq","2nd Mode %"]
        continuous_columns = self.categorical;
        self.__all_categorical_table = collections.OrderedDict();
        for col, cat in enumerate(continuous_columns.keys()):

            dataFeature = self.fileCSV[cat];
            hasInt = False;
            countWrongItem = 0;

            for value in dataFeature:
                if type(value) is int:
                    hasInt = True

                elif "Not in universe" in value:
                    countWrongItem = countWrongItem + 1 
            if hasInt is False :
                
                if countWrongItem/ dataFeature.size * 100 < 60:
                    feature = collections.OrderedDict();
    #           Put in the table feature the name of the feature
                    feature['nameFeature'] = cat
    #           Put in the table feature the total count of lines
                    feature['countTotal'] = dataFeature.size
                    feature['% Miss'] = countWrongItem/ dataFeature.size * 100
                    feature['cardTotal'] = np.unique(dataFeature).size
            
    #           check 
                    iFirstMode= 0;
                    while("Not in universe" in dataFeature.value_counts().keys()[iFirstMode] or "?" in dataFeature.value_counts().keys()[iFirstMode]):
                        iFirstMode = iFirstMode + 1
                
                    iSecondMode= iFirstMode + 1;
                    while("Not in universe" in dataFeature.value_counts().keys()[iSecondMode] or "?" in dataFeature.value_counts().keys()[iSecondMode]):
                        iSecondMode = iSecondMode + 1
                        
                    feature['First Mode'] = dataFeature.value_counts().keys()[iFirstMode]
                    feature['First Mode Freq'] = dataFeature.value_counts()[iFirstMode]
                    feature['First Mode %'] = round(dataFeature.value_counts()[iFirstMode] / dataFeature.size * 100,2)
                    
                    feature['Second Mode'] = dataFeature.value_counts().keys()[iSecondMode]
                    feature['Second Mode Freq'] = dataFeature.value_counts()[iSecondMode]
                    feature['Second Mode %'] = round(dataFeature.value_counts()[iSecondMode] / dataFeature.size * 100,2)
                    self.__continuous_features_table.append(feature);
                    
                    self.__all_categorical_header.append(cat);
                    self.__all_categorical_table[cat] = continuous_columns.values[:,col];
                  
#        Write DQR continuous
        self.write_results(self.__continuous_features_table, self.__categorical_header, self.pathDQR);
        self.write_results(self.__all_categorical_table, self.__all_categorical_header,"./results/categorical-features.csv");
