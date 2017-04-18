from data_reader import *


# return dataframe with all n/a filled with column means
def df_fill_na(df,method="mean"):
  if method == 'mean':
    mean_vals = df.mean(axis=0)
    print 'mean_vals are', mean_vals
    filled_df = df.fillna(df.mean())
    return filled_df

# return dataframe with lag features
# more importantly, add NaN correspondingly since obs from different ids are stacked together
# input: df - pandas.DataFrame, features - list of str, lags - list of lag periods, all should be positive values
def df_add_lag_features(df, features=["y","y"], lags=[1,2]):
  assert len(features) == len(lags)
  ids = get_all_ids(df)
  first_occur_index_dict = {id_:-1 for id_ in ids}
  for id_ in ids:
    first_occur_index_dict[id_] = df[df.id==id_].index.tolist()[0]
  for i, feature_name in enumerate(features):
    assert feature_name in df.columns
    lag = lags[i]
    lag_feature_name="%s_lag%s"%(feature_name,lag)
    assert lag_feature_name not in df.columns
    df = df.assign(lag_feature_name=df[feature_name].shift(lag).values)
    df = df.rename(columns={'lag_feature_name':lag_feature_name})
    for id_ in ids:
      idx = first_occur_index_dict[id_]
      df.ix[idx:idx+lag-1,lag_feature_name] = None # both start and stop bounds are INCLUDED, see http://pandas.pydata.org/pandas-docs/stable/indexing.html
  return df


# return dataframe where rows with NaN values are removed. This method is called after lagged variables added
def df_drop_rows_with_na(df):
    dropped = df.dropna(axis=0)
    dropped.reset_index(inplace=True, drop=True)
    return dropped


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
  df = df_fill_na(df)
  df = df_add_lag_features(df,features=["y","y"], lags=[1,2]) # add y(t-1) and y(t-2) currently
  #df = df_drop_rows_with_na(df) 


  return df




if __name__ == '__main__':
  main()
