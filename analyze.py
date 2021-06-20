import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

class analyze:
    def __init__(self):
        self.suggest_read_csv()


    def suggest_read_csv(self, path='/kaggle/input'):
        for dir_name, _, file_names in os.walk('/kaggle/input'):
            for file_name in file_names:
                file = file_name.split('.')
                if file[-1] == 'csv':
                    print(f"{file[0]} = pd.read_csv('{os.path.join(dir_name, file_name)}')")

    def is_display_or_return(self, obj, is_display):
        if is_display:
            display(obj)
            return 0
        else:
            return obj

    def get_dtypes(self, dataframe, display=False):
        dtypes_df = pd.DataFrame(dataframe.dtypes).reset_index()
        dtypes_df.columns = ['COLUMN','DTYPE']
        dtypes_df['COLS'] = dtypes_df.groupby(['DTYPE'])['COLUMN'].transform(lambda x: ', '.join(x))
        dtypes_df = dtypes_df[['DTYPE','COLS']].drop_duplicates().reset_index(drop=True)
        return is_display_or_return(dtypes_df, display)
        
    def get_head_tail(self, dataframe, display=False, count=3):
        head_tail_df = pd.concat([dataframe.head(count),dataframe.tail(count)])
        return is_display_or_return(head_tail_df, display)


    def get_col_stats(self, dataframe, display=False):
        from functools import reduce
        data_frames = [pd.DataFrame(dataframe.dtypes,columns=['DTYPES']),
                       pd.DataFrame(dataframe.isnull().sum(),columns=['IS_NULL']),
                       dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T]
        col_stats_df = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'), data_frames)
        return is_display_or_return(col_stats_df,display)

    def define_df_cols(self, dataframe, cat_th=10, car_th=20, is_display=False):

        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat_cols = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car_cols = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

        cat_cols = cat_cols + num_but_cat_cols
        cat_cols = [col for col in cat_cols if col not in cat_but_car_cols]

        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat_cols]

        stats = pd.DataFrame(index=['Categoric Cols','Numeric Cols','Categoric But Car Cols','Numeric But Categoric Cols'])
        stats['Count'] = [len(cat_cols), len(num_cols), len(cat_but_car_cols), len(num_but_cat_cols)]
        stats['Columns'] = [", ".join(cat_cols), ", ".join(num_cols), ", ".join(cat_but_car_cols), ", ".join(num_but_cat_cols)]
        
        if is_display:
            display(stats)
        
        return cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols


    def plot_cols(self, dataframe, col_name, kind='bar'):
        plt.figure(figsize=(10,6))
        
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


    def cat_summary(self, dataframe, col_name, threshold=False, display=False, plot=False):
        df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),"Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
        if threshold!=False:
            threshold_column = col_name if threshold > 1 else 'Ratio'
            if threshold <=1:
                  df[threshold_column] = df[threshold_column]/100 
            main_df = df[df[threshold_column]>threshold]
            th_df = pd.DataFrame(df[df[threshold_column]<threshold].sum()).T
            th_df.index = ['Others']
            df = pd.concat([main_df,th_df])
        
        if plot:
            plot_cols(df,col_name)
            
        return is_display_or_return(df,display)


    def num_summary(self, dataframe, col_name, display=False, plot=False,
                   quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]):

        df = pd.DataFrame(dataframe[col_name].describe(quantiles).T)
        
        if plot:
            plot_cols(dataframe,col_name,'hist')
            
        return is_display_or_return(df,display)
        

    def col_target_summary(self, dataframe, target, col_name, display=False, plot=False):
        cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols = define_df_cols(dataframe)
        
        if col_name in cat_cols+num_but_cat_cols:
            dataframe = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col_name)[target].mean()})
        elif col_name in num_cols+cat_but_car_cols:
            dataframe = dataframe.groupby(target).agg({col_name: "mean"}).sort_values(by=col_name,ascending=False)
        
        if plot:
            plot_cols(dataframe,'TARGET_MEAN')
            
        return is_display_or_return(dataframe, display)
        
                
analyzer = analyze()

sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

target_column = 'SalePrice'

_ = get_dtypes(train,False)
_ = get_dtypes(test,False)

_ = get_head_tail(train,False, 2)
_ = get_head_tail(test,False, 2)

_ = get_col_stats(train,False)
_ = get_col_stats(test,False)

cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols = define_df_cols(train, is_display=True)

_ = cat_summary(train,'LotShape', threshold=70, display=True, plot=True)

_ = num_summary(train,'MSSubClass',quantiles=[0.2,0.5], display=False, plot=True)

_ = col_target_summary(train, target_column, 'LotShape', display=False, plot=True)
_ = col_target_summary(train, target_column, 'Alley', display=False, plot=True)