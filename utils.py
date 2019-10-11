from collections import Counter
import pandas as pd
import plotly as ply

class Utils:
    
    def getHeader():
        return {
				'AAGE': 'age',
				'ACLSWKR': 'class of worker',
				'ADTIND': 'industry code',
				'ADTOCC': 'occupation code',
				'AGI': 'adjusted gross income',
				'AHGA': 'education',
				'AHRSPAY': 'wage per hour',
				'AHSCOL': 'enrolled in edu inst last wk',
				'AMARITL': 'marital status',
				'AMJIND': 'major industry code',
				'AMJOCC': 'major occupation code',
				'ARACE': 'mace',
				'AREORGN': 'hispanic Origin',
				'ASEX': 'sex',
				'AUNMEM': 'member of a labor union',
				'AUNTYPE': 'reason for unemployment',
				'AWKSTAT': 'full or part time employment stat',
				'CAPGAIN': 'capital gains',
				'CAPLOSS': 'capital losses',
				'DIVVAL': 'divdends from stocks',
				'FEDTAX': 'federal income tax liability',
				'FILESTAT': 'tax filer status',
				'GRINREG': 'region of previous residence',
				'GRINST': 'state of previous residence',
				'HHDFMX': 'detailed household and family stat',
				'HHDREL': 'detailed household summary in household',
				'MARSUPWT': 'instance weight',
				'MIGMTR1': 'migration code-change in msa',
				'MIGMTR3': 'migration code-change in reg',
				'MIGMTR4': 'migration code-move within reg',
				'MIGSAME': 'live in this house 1 year ago',
				'MIGSUN': 'migration prev res in sunbelt',
				'NOEMP': 'num persons worked for employer',
				'PARENT': 'family members under 18',
				'PEARNVAL': 'total person earnings',
				'PEFNTVTY': 'country of birth father',
				'PEMNTVTY': 'country of birth mother',
				'PENATVTY': 'country of birth self',
				'PRCITSHP': 'citizenship',
				'PTOTVAL': 'total person income',
				'SEOTR': 'own business or self employed',
				'TAXINC': 'taxable income amount',
				'VETQVA': 'fill inc questionnaire for veterans admin',
				'VETYN': 'veterans benefits',
				'WKSWORK': 'weeks worked in year'
			}

		
    def graph_continuous(self):
        'raw = self.get_continuous()'
        df = pd.read_csv("./results/E-DQR-continuous.csv")
        dc = pd.read_csv("./results/continuous-features.csv")
        'print(df)'
        size = df.shape 
        data = {col: list(df[col]) for col in df.columns}
        data_continuous = {col: list(dc[col]) for col in dc.columns}
        
        for i in range(0,size[0]):
            feature = data["Feature name"][i]
                
            ply.offline.plot({
                "data": [ply.graph_objs.Box(x=data_continuous[feature])],
                "layout": ply.graph_objs.Layout(
                            title="Box plot for feature : '" + feature + "' to check outliers "
                        )
            }, filename="./results/graphs/%s.html" %feature+ "possible_out")
                                    
            if data["Card"][i] >=10:
                
                ply.offline.plot({
                      "data": [ply.graph_objs.Histogram(x=data_continuous[feature])],
                      "layout": ply.graph_objs.Layout(
                              title="Histogram of feature : '" + feature + "' for cardinality >=10"
                              )
                }, filename="./results/graphs/%s.html" %feature)
                
            else:
                tab_value = Counter(data_continuous[feature])
                val = []
                kley = []
                for cle, valeur in tab_value.items():
#                    print("La clé {} contient la valeur {}.".format(cle, valeur))
                    val.append(valeur)
                    kley.append(cle)
                print(cle,valeur)
# =============================================================================
#                 tab_value = Counter(data_continuous[feature])
#                 print(tab_value.keys())
# =============================================================================
                
                ply.offline.plot({
                        "data": [ply.graph_objs.Bar(x=kley, y=val)],
                        "layout": ply.graph_objs.Layout(
                                    title="Bar plot for feature : '" + feature + "' for cardinality <10"
                                )
                }, filename="./results/graphs/%s.html" % feature)
                        
    def graph_categorical(self):
            
        df = pd.read_csv("./results/E-DQR-categorical.csv");
        dc = pd.read_csv("./results/categorical-features.csv");
        size = df.shape;
        
        data = {col: list(df[col]) for col in df.columns}
        data_categorical = {col: list(dc[col]) for col in dc.columns}
        
        for i in range(0,size[0]):
            feature = data["Feature name"][i]
            tab_value = Counter(data_categorical[feature])
            val = []
            kley = []
            for cle, valeur in tab_value.items():
                
                val.append(valeur)
                kley.append(cle)
    
            
            ply.offline.plot({
                    "data": [
                            ply.graph_objs.Bar(
                                    x=kley,
                                    y=val
                                    )
                            ],
                    "layout": ply.graph_objs.Layout(
                            title="Bar plot for feature : '" + feature + "' for categorical"
                            )
                    }, filename="./results/graphs/%s.html" % feature)
        