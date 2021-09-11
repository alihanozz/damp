class analyzer:
    def __init__(self, display_=True, plot_=True, data='/kaggle/input', sep_=','):
        self.check_modules()
        self.import_libraries()
        self.df_list = []
        self.data = None
        self.read_files(data, sep_)
        self.display_ = display_
        self.plot_ = plot_

    def check_modules(self, modules={'pandas','numpy','matplotlib','seaborn','sklearn'}):
        print('Checking modules...')
        self.global_imports('sys')
        self.global_imports('subprocess')
        self.global_imports('pkg_resources')

        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = modules - installed

        if missing:
            self.install_modules(missing)

    def install_modules(self, modules):
        print('installing modules:'+' '.join(modules))
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *modules], stdout=subprocess.DEVNULL)

    def global_imports(self, modulename, shortname = None):
        if shortname is None: 
            shortname = modulename

        print("\t","Import:",modulename,"as",shortname)
        if '.' not in modulename:
            globals()[shortname] = __import__(modulename)
        else:
            *modulename1, modulename2 = modulename.split('.')
            modulename1 = ".".join(modulename1)
            self.global_imports(modulename1)
            globals()[shortname] = eval(modulename1 + "." + modulename2)

    def import_libraries(self):
        print('Importing libraries...')
        
        self.global_imports('numpy','np')
        self.global_imports('pandas','pd')
        self.global_imports('os')
        self.global_imports('seaborn','sns')
        sns.axes_style("whitegrid")
        self.global_imports('matplotlib.pyplot','plt')
        self.global_imports('datetime.datetime','datetime')
        self.global_imports('functools.reduce','reduce')
        self.global_imports('warnings')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 50)
        pd.set_option('max_colwidth', 100)

    def read_files(self, data, sep_):
        print('Reading files...')
        if type(data)!=str:
            self.data = data
            exec(f"self.df_list.append('data')")
        else:
            ext_dict = {'csv':'csv','xlsx':'excel','pickle':'pickle'}
            for dir_name, _, file_names in os.walk(data):
                for file_name in file_names:
                    file = file_name.split('.')
                    file_ext = file[-1]
                    if  file_ext in ext_dict.keys():
                        if file_ext == 'xlsx':
                            self.check_modules({'openpyxl','xlrd'})
                        sep_str = f", sep='{sep_}'" if sep_!=',' else ''
                        command = f"self.{file[0]} = pd.read_{ext_dict[file_ext]}('{os.path.join(dir_name, file_name)}'"+sep_str+f")"
                        exec(command)
                        if file_ext != 'pickle':
                            exec(f"self.{file[0]} = self.reduce_mem_usage(self.{file[0]})")
                        exec(f"self.df_list.append('{file[0]}')")
                        print(f"\t {file[0]} is created.")
                        print()

    def save_dfs(self, path='/kaggle/output'):
        for d in self.df_list:
            full_path = os.path.join(path, d+'.pickle')
            exec(f"self.{d}.to_pickle('{full_path}')")
            print(f"{d} saved to {full_path}")

    def rename_dfs(self, rename_dict={}):
        self.global_imports('gc')
        for old, new in rename_dict.items():
            exec(f"self.{new} = self.{old}.copy()")
            exec(f"del self.{old}")
            exec("gc.collect()")
            exec(f"self.df_list.append('{new}')")
            exec(f"self.df_list.remove('{old}')")

    def set_dataframe(self, dataframe):
        if len(dataframe) == 0:
            dataframe = self.data.copy(deep=False) if self.data is not None else dataframe
        return dataframe

    def dtypes(self, dataframe=[], display_=None):
        dataframe = self.set_dataframe(dataframe)
        dtypes_df = pd.DataFrame(dataframe.dtypes).reset_index()
        dtypes_df.columns = ['COLUMN','DTYPE']
        dtypes_df['COLS'] = dtypes_df.astype(str).groupby(['DTYPE'])['COLUMN'].transform(lambda x: ', '.join(x))
        dtypes_df = dtypes_df[['DTYPE','COLS']].reset_index(drop=True).set_index('DTYPE').drop_duplicates()
        
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(dtypes_df)
            
        return dtypes_df
        
    def head_tail(self, dataframe=[], count=3, display_=None):
        dataframe = self.set_dataframe(dataframe)
        head_tail_df = pd.concat([dataframe.head(count),dataframe.tail(count)])
        
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(head_tail_df)
            
        return head_tail_df
    
    def drop_unnecessary_cols(self, dataframe=[], cols=[]):
        col_stats = self.col_stats(dataframe, display_=False);
        cols_to_drop_lst = col_stats.NUNIQUE[col_stats.NUNIQUE<=1].keys().to_list()
        return dataframe.drop(cols_to_drop_lst+cols, axis=1)

    def col_stats(self, dataframe=[], display_=None):
        dataframe = self.set_dataframe(dataframe)
        data_frames = [pd.DataFrame(dataframe.dtypes,columns=['DTYPES']),
                       pd.DataFrame(dataframe.isnull().sum(),columns=['IS_NULL']),
                       pd.DataFrame(dataframe.nunique(),columns=['NUNIQUE']),
                       dataframe.quantile([0, 0.01, 0.05, 0.50, 0.95, 0.99, 1]).T]
        col_stats_df = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'), data_frames)
        
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(col_stats_df)
            
        return col_stats_df

    def df_cols(self, dataframe=[], cat_th=10, car_th=20, id_th=100, display_=None):
        dataframe = self.set_dataframe(dataframe)

        numeric_cols   = dataframe._get_numeric_data().columns
        
        nunique        = dataframe.nunique()
        df_unique_pct  = nunique/dataframe.shape[0]
        self.id_cols   = list(set(list(df_unique_pct[(df_unique_pct>0.5)].keys()) + [col for col in list(nunique[nunique>id_th].keys()) if col[-2:].lower() == 'id']))
        
        self.cat_cols  = list(dataframe.select_dtypes(["category","object"]).columns)
        self.cat_but_car_cols = [col for col in self.cat_cols if (dataframe[col].nunique() > car_th) and (col not in self.id_cols)]
        self.num_but_cat_cols = [col for col in numeric_cols  if (dataframe[col].nunique() < cat_th) and (col not in self.id_cols)]
        self.num_but_cat_but_car_cols = [col for col in self.num_but_cat_cols if (dataframe[col].nunique() > car_th) and (col not in self.id_cols)]
        self.num_but_cat_cols = [col for col in self.num_but_cat_cols  if col not in self.num_but_cat_but_car_cols]
        
        self.cat_cols  = [col for col in self.cat_cols if col not in self.cat_but_car_cols+self.id_cols]
        self.num_cols  = [col for col in numeric_cols  if col not in self.num_but_cat_cols+self.num_but_cat_but_car_cols+self.id_cols]

        self.date_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "datetime64[ns]"]        
        self.id_cols   = [col for col in self.id_cols if col not in self.date_cols]
        

        stats = pd.DataFrame(index=['ID Cols', 'Date Cols', 'Numeric Cols', 'Categoric (Low Cardinality) Cols', 
                                    'Categoric (High Cardinality) Cols', 'Numeric But Categoric (Low Cardinality) Cols',  
                                    'Numeric But Categoric (High Cardinality) Cols'])
        stats['Count'] = [len(self.id_cols), len(self.date_cols), len(self.num_cols), len(self.cat_cols), \
                          len(self.cat_but_car_cols), len(self.num_but_cat_cols), len(self.num_but_cat_but_car_cols)]
        stats['Columns'] = [", ".join(self.id_cols), ", ".join(self.date_cols), ", ".join(self.num_cols), ", ".join(self.cat_cols), \
                            ", ".join(self.cat_but_car_cols), ", ".join(self.num_but_cat_cols), ", ".join(self.num_but_cat_but_car_cols)]
        
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(stats)
        
        return self.id_cols, self.date_cols, self.num_cols, self.cat_cols, self.cat_but_car_cols, self.num_but_cat_cols, self.num_but_cat_but_car_cols

    def plot_cols(self, dataframe=[], col_name=None, kind='bar', palette='Set2'):
        dataframe = self.set_dataframe(dataframe)
        plt.figure(figsize=(8,5))
        
        if kind == 'hist':
            sns.set_palette(sns.color_palette(palette))
            ax = sns.histplot(dataframe[col_name], bins=20, kde=True)
            plt.xlabel('')
        if kind == 'bar':
            sns.set_palette(sns.color_palette(palette))
            dataframe.index.names = ['index']
            ax = sns.barplot(x='index', y=col_name, data=dataframe.reset_index())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            plt.xlabel('')

        plt.title(col_name)
        plt.tight_layout()
        plt.show()

    def cat_summ(self, dataframe=[], col_name=None, threshold=False, display_=None, plot_=None):
        dataframe = self.set_dataframe(dataframe)
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
        
        plot_ = self.plot_ if plot_ is None else plot_
        if plot_:
            self.plot_cols(cat_summary_df,col_name)
        
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(cat_summary_df)
            
        return cat_summary_df

    def num_summ(self, dataframe=[], col_name=None, display_=None, plot_=None,
                   quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]):

        dataframe = self.set_dataframe(dataframe)
        num_summary_df = pd.DataFrame(dataframe[col_name].describe(quantiles).T)
        
        plot_ = self.plot_ if plot_ is None else plot_
        if plot_:
            self.plot_cols(dataframe,col_name,'hist',"rainbow_r")
            
        if display_:
            display(num_summary_df)
            
        return num_summary_df
        
    def col_summ(self, dataframe=[], target=None, col_name=None, display_=None, plot_=None):
        dataframe = self.set_dataframe(dataframe)
        cat_cols, num_cols, cat_but_car_cols, num_but_cat_cols, date_cols = self.df_cols(dataframe)
        
        if col_name in cat_cols+num_but_cat_cols:
            col_target_summary_df = pd.DataFrame({col_name: dataframe.groupby(col_name)[target].mean()})
        elif col_name in num_cols+cat_but_car_cols:
            col_target_summary_df = dataframe.groupby(target).agg({col_name: "mean"}).sort_values(by=col_name,ascending=False)
        
        plot_ = self.plot_ if plot_ is None else plot_
        if plot_:
            self.plot_cols(col_target_summary_df,col_name)
            
        display_ = self.display_ if display_ is None else display_
        if display_:
            display(col_target_summary_df)
            
        return col_target_summary_df
    
    def reduce_mem_usage(self, df):
        start_mem = df.memory_usage().sum() / 1024**2
        print('\t Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == object:
                if df[col].nunique()<300:
                    df[col] = df[col].astype('category')
                    continue
            if str(col_type)[:3] in ['int','flo']:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_mem = df.memory_usage().sum() / 1024**2
        print('\t Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('\t Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
            
        return df