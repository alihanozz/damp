import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

class analyze:
    def __init__(self, target_column='', is_display=True, is_plot=True):
        self.dataframe_list = []
        self.target_column = target_column
        self.is_display = is_display
        self.is_plot = is_plot
        self.read_csv_files()
    
    def read_csv_files(self, path='/kaggle/input'):
        for dir_name, _, file_names in os.walk('/kaggle/input'):
            for file_name in file_names:
                file = file_name.split('.')
                if file[-1] == 'csv':
                    exec(f"self.{file[0]} = pd.read_csv('{os.path.join(dir_name, file_name)}')")
                    self.dataframe_list.append(file[0])
                    print(f"{file[0]} is created.")


    def get_dtypes(self, dataframe, is_display=None):
        dtypes_df = pd.DataFrame(dataframe.dtypes).reset_index()
        dtypes_df.columns = ['COLUMN','DTYPE']
        dtypes_df['COLS'] = dtypes_df.groupby(['DTYPE'])['COLUMN'].transform(lambda x: ', '.join(x))
        dtypes_df = dtypes_df[['DTYPE','COLS']].drop_duplicates().reset_index(drop=True).set_index('DTYPE')
        
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(dtypes_df)
            
        return dtypes_df
        
    def get_head_tail(self, dataframe, count=3, is_display=None):
        head_tail_df = pd.concat([dataframe.head(count),dataframe.tail(count)])
        
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(head_tail_df)
            
        return head_tail_df


    def get_col_stats(self, dataframe, is_display=None):
        data_frames = [pd.DataFrame(dataframe.dtypes,columns=['DTYPES']),
                       pd.DataFrame(dataframe.isnull().sum(),columns=['IS_NULL']),
                       dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T]
        col_stats_df = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'), data_frames)
        
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(col_stats_df)
            
        return col_stats_df

    def define_df_cols(self, dataframe, cat_th=10, car_th=20, is_display=None):

        self.cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        self.num_but_cat_cols = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        self.cat_but_car_cols = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

        self.cat_cols = self.cat_cols + self.num_but_cat_cols
        self.cat_cols = [col for col in cat_cols if col not in self.cat_but_car_cols]

        self.num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        self.num_cols = [col for col in self.num_cols if col not in self.num_but_cat_cols]

        stats = pd.DataFrame(index=['Categoric Cols','Numeric Cols','Categoric But Car Cols','Numeric But Categoric Cols'])
        stats['Count'] = [len(self.cat_cols), len(self.num_cols), len(self.cat_but_car_cols), len(self.num_but_cat_cols)]
        stats['Columns'] = [", ".join(self.cat_cols), ", ".join(self.num_cols), ", ".join(self.cat_but_car_cols), ", ".join(self.num_but_cat_cols)]
        
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(stats)
        
        return self.cat_cols, self.num_cols, self.cat_but_car_cols, self.num_but_cat_cols


    def plot_cols(self, dataframe, col_name, kind='bar'):
        plt.figure(figsize=(8,5))
        
        if kind == 'hist':
            ax = sns.histplot(dataframe[col_name], bins=20)
            plt.xlabel('')
        if kind == 'bar':
            dataframe.index.names = ['index']
            ax = sns.barplot(x='index', y=col_name, data=dataframe.reset_index())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            plt.xlabel('')

        plt.title(col_name)
        plt.tight_layout()
        plt.show()

    def cat_summary(self, dataframe, col_name, threshold=False, is_display=None, is_plot=None):
        df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),"Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
        
        if threshold!=False:
            threshold_column = col_name if threshold > 1 else 'Ratio'
            if threshold <=1:
                  df[threshold_column] = df[threshold_column]/100 
            main_df = df[df[threshold_column]>threshold]
            th_df = pd.DataFrame(df[df[threshold_column]<threshold].sum()).T
            th_df.index = ['Others']
            cat_summary_df = pd.concat([main_df,th_df])
        else:
            cat_summary_df = df
        
        is_plot = self.is_plot if is_plot is None else is_plot
        if is_plot:
            self.plot_cols(cat_summary_df,col_name)
        
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(cat_summary_df)
            
        return cat_summary_df

    def num_summary(self, dataframe, col_name, is_display=None, is_plot=None,
                   quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]):

        num_summary_df = pd.DataFrame(dataframe[col_name].describe(quantiles).T)
        
        is_plot = self.is_plot if is_plot is None else is_plot
        if is_plot:
            self.plot_cols(dataframe,col_name,'hist')
            
        if is_display:
            display(num_summary_df)
            
        return num_summary_df
        
    def col_target_summary(self, dataframe, target, col_name, is_display=None, is_plot=None):
        cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols = self.define_df_cols(dataframe)
        
        if col_name in cat_cols+num_but_cat_cols:
            col_target_summary_df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col_name)[target].mean()})
        elif col_name in num_cols+cat_but_car_cols:
            col_target_summary_df = dataframe.groupby(target).agg({col_name: "mean"}).sort_values(by=col_name,ascending=False)
        
        is_plot = self.is_plot if is_plot is None else is_plot
        if is_plot:
            self.plot_cols(col_target_summary_df,col_name)
            
        is_display = self.is_display if is_display is None else is_display
        if is_display:
            display(col_target_summary_df)
            
        return col_target_summary_df
    
                
analyzer = analyze('SalePrice', is_display=True, is_plot=True)

train_dtypes = analyzer.get_dtypes(analyzer.train);
test_dtypes = analyzer.get_dtypes(analyzer.test, is_display=False); 

analyzer.get_head_tail(analyzer.train, 2);
analyzer.get_head_tail(analyzer.test, 2, is_display=False);

train_col_stats = analyzer.get_col_stats(analyzer.train);
test_col_stats = analyzer.get_col_stats(analyzer.test, is_display=False);

cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols = analyzer.define_df_cols(analyzer.train, cat_th=10, car_th=20);
cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols = analyzer.define_df_cols(analyzer.train, is_display=False);

for cat_col in cat_cols:
    analyzer.cat_summary(train, cat_col, threshold=1000, is_display=False);
    analyzer.cat_summary(train, cat_col, threshold=0.3, is_plot=False);

for num_col in num_cols:
    analyzer.num_summary(train, num_col, quantiles=[0.2,0.5], is_display=False)
    analyzer.num_summary(train, num_col, is_plot=False)

for col in train.columns:
    if train[col].nunique()<=20:
        analyzer.col_target_summary(train, analyzer.target_column, col, is_display=False, is_plot=False)
