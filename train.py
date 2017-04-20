from data_reader import *


# return dataframe with all n/a filled with column means
def df_fill_na(df,method="mean"):
  if method == 'mean':
    mean_vals = df.mean(axis=0)
    filled_df = df.fillna(df.mean())
    return filled_df

# derived_17 and fundamental_17's mean value are 7.736514e+11 and 7.093846e+13.
def df_drop_extreme_value_columns(df):
  df.drop('derived_1', axis=1, inplace=True)
  df.drop('fundamental_17', axis=1, inplace=True)
  return df

# return dataframe with lag features
# more importantly, add NaN correspondingly since obs from different ids are stacked together
# input: df - pandas.DataFrame, features - list of str, lags - list of lag periods, all should be positive values
def df_add_lag_features(df, features=["y","y"], lags=[1,2]):
  assert len(features) == len(lags)
  
  for i, feature_name in enumerate(features):
    print 'i=%d,total=%d'%(i, len(features))
    assert feature_name in df.columns
    lag = lags[i]
    lag_feature_name="%s_lag%s"%(feature_name,lag)
    assert lag_feature_name not in df.columns
    df = df.assign(lag_feature_name=df[feature_name].shift(lag).values)
    df = df.rename(columns={'lag_feature_name':lag_feature_name})
    
  return df


# return dataframe where rows should contain NaN values are removed. This method is called after lagged variables added.
# However, for efficiency reason, NaNs are not added but found by computing first occurred index
def df_drop_rows_with_na(df,max_lag=1):
    ids = get_all_ids(df)
    first_occur_index_dict = {id_:-1 for id_ in ids}
    labels_to_drop = []
    for id_ in ids:
      first_occur_index = df[df.id==id_].index.tolist()[0]
      labels_to_drop.extend([first_occur_index+i for i in range(max_lag)])
    
    dropped = df.drop(labels=labels_to_drop)
    dropped.reset_index(inplace=True, drop=True)
    return dropped

# output df.columns.tolist()=[id,timestamp, sorted(features), y1]
def df_reorder_columns(df):
  feature_cols = sorted(list(set(df.columns.values.tolist()) - set(['id','timestamp','y1'])))
  reordered_cols = ['id','timestamp'] + feature_cols + ['y1']
  return df[reordered_cols]

def df_reorder_rows(df):
  df = df.sort_values(by=['timestamp','id']) # inplace=True will cause warning
  return df

# return two dataframe namely first and second
# where first contains obs that ts<=split_timestamp, second contains ts>split_timestamp
def df_split_by_timestamp(df, split_timestamp=1500):
    first_df = df[df.timestamp < split_timestamp]
    second_df = df[df.timestamp >= split_timestamp]
    return first_df, second_df


def main():
  '''
  1. create dataframe df by loading data from database, df has NaN entries
  2. fill df NaN with mean values
  3. add lagged features (this will create new NaN entries due to lagging variables)
  4. remove rows with NaN entries
  5. split df into df_train, df_test by timestamp=1500, hardcoded
  6. build two dataframe from dt_test, namely df_test_features(. - y1 + y1_est) and df_test_y1(id,timestamp,y1), where y1_est will be filled with estimated y1 values
  7. we could save checkpoint file, ckpt={df_train, df_test_features, df_test_y1}
  8. ranking attributes by correlation on df_train
  9. Fa's Plan Search here, got best model, save ckpt2=model_parameters
  10. compute mean square error based on y1_est and y1
  Note: step 3 and 9 are conflicting now
  '''
  df = import_df_from_csv("preprocessed_data.csv") # this will be replaced by querying database
  df = df_drop_extreme_value_columns(df)
  df = df_fill_na(df)
  feature_names = ['derived_0', 'fundamental_0', \
   'fundamental_7',  'fundamental_18', \
   'fundamental_19', 'fundamental_21', 'fundamental_33', \
   'fundamental_36', 'fundamental_41', 'fundamental_42', \
   'fundamental_45', 'fundamental_48', 'fundamental_53', \
   'fundamental_59', 'technical_0', 'technical_2', 'technical_3', \
   'technical_6', 'technical_7', 'technical_9', 'technical_11', \
   'technical_12', 'technical_13', 'technical_14', 'technical_16', \
   'technical_17', 'technical_18', 'technical_19', 'technical_20', \
   'technical_21', 'technical_22', 'technical_24', 'technical_27', \
   'technical_29', 'technical_30', 'technical_32', 'technical_33', \
   'technical_34', 'technical_35', 'technical_36', 'technical_37', \
   'technical_38', 'technical_39', 'technical_40', 'technical_41', \
   'technical_42', 'technical_43', 'y']
  feature_list = []
  lag_list = []
  for feature in feature_names:
    feature_list.extend([feature])
    lag_list.extend([1])

  df = df_add_lag_features(df,features=feature_list, lags=lag_list) # add t-1,t-2  lagged terms
  df = df_drop_rows_with_na(df, max_lag=max(lag_list))
  df = df_reorder_rows(df)
  df = df_reorder_columns(df)
  return df

  



if __name__ == '__main__':
  main()
