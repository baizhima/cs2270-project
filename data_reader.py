# CS 2270 Project
# Yahui Wang, Qian Mei and Shan Lu

import pandas as pd

# data exploration
def df_explore(df):
    print 'training data exploration'
    print 'all ids are ', get_all_ids(df)
    print 'all timestamps are', get_all_timestamps(df)
    print 'total number of observations', len(df) # ~1710000 obs
    
    print 'security ids with at least 100 obs are', get_fruitful_ids(df)
    print 'features with at most 100,000 n/a values are', get_fruitful_columns(df)


# return a sorted list of security ids, total 1424, range in [0,2158]
def get_all_ids(df):
    idlst = list(df['id'].unique())
    idlst.sort()
    return idlst

# return a sorted list of timestamps 0-1812
def get_all_timestamps(df):
    tslst = list(df['timestamp'].unique())
    tslst.sort()
    return tslst

# return a dataframe of particular security id
def get_obs_by_id(df, id):
    return df[df.id == id]

# return a dataframe of particular timestamp
def get_obs_by_timestamp(df, timestamp):
    return df[df.timestamp == timestamp] 

# return columns that have less than 'max_na_count' n/a values
def get_fruitful_columns(df, max_na_count=100000):
    col_names = list(df)
    nacounts = df.isnull().sum()
    fruitful_cols = []
    for col in col_names:
        if nacounts[col]<=max_na_count:
            fruitful_cols.append(col)
    return fruitful_cols

# return a list of id that have at least min_num_obs observations
def get_fruitful_ids(df, min_num_obs=100):
    ids = get_all_ids(df)
    idcounts = pd.value_counts(df['id'].values, sort=True)
    fruitful_ids = []
    for id_ in ids:
        if idcounts[id_] >= min_num_obs:
            fruitful_ids.append(id_)
    return fruitful_ids

# return a dataframe where all ids have at least 100 observations
# selected features are those with less than 100000 n/a enries
# n/a will be maintained here, might be filled with mean column values during model training
def build_working_df(df):
    fruitful_ids = get_fruitful_ids(df)
    cols = get_fruitful_columns(df)
    print 'building working dataframe...'
    print 'Step 1: column selection'
    df1 = df[cols]
    print 'Step 2: id selection'
    df2 = df1[df1['id'].isin(fruitful_ids)]
    print 'Step 3: sort the df by (id, timestamp) incrementally, reset dataframe index by new order'
    df2=df2.sort_values(by=['id','timestamp'])
    df2.reset_index(inplace=True, drop=True)
    print 'Step 4: add y(t+1) by shifting column y'
    df3 = df2.assign(y1=df2.y.shift(-1).values)
    print 'Step 5: assign NaN to last occurrence of all ids y(t+1) because obs from different ids are stacked together'
    for i in range(1,len(fruitful_ids)):
        id_ = fruitful_ids[i]
        first_occur_idx = df3[df3.id==id_].index.tolist()[0]
        print 'id=%d,first_occur_idx=%d'%(id_, first_occur_idx)
        df3.ix[first_occur_idx-1,'y1'] = None
    return df3



# return dataframe with all n/a filled with column means
def df_fill_na(df):
    mean_vals = df.mean(axis=0)
    print 'mean_vals are', mean_vals
    filled_df = df.fillna(df.mean())
    return filled_df



# return two dataframe namely first and second
# where first contains obs that ts<=split_timestamp, second contains ts>split_timestamp
def split_df_by_timestamp(df, split_timestamp=1500):
    first_df = df[df.timestamp < split_timestamp]
    second_df = df[df.timestamp >= split_timestamp]
    return first_df, second_df



# export pre-processed dataframe into a csv file
def export_df_to_csv(df, csv_filename="preprocessed_data.csv"):
    print 'exporting dataframe to ', csv_filename
    df.to_csv(csv_filename,sep=',',na_rep='',float_format="%.8f",index=False)

def import_df_from_csv(csv_filename="preprocessed_data.csv"):
    print 'importing dataframe from ', csv_filename
    df = pd.read_csv(csv_filename,sep=',',index_col=None)
    return df

if __name__ == '__main__':
    print 'Hello'
    with pd.HDFStore("train.h5", "r") as train:
        df = train.get("train")
        df_explore(df)
        work_df = build_working_df(df)
        export_df_to_csv(work_df)
        work_df_no_na = df_fill_na(work_df)
        export_df_to_csv(work_df_no_na, "preprocessed_data_no_na.csv")
