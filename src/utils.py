import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



def get_series(df, series_id):
    """
    return portion of data corresponding to series_id
    DataFrame * str -> DataFrame
    """
    return df[df["series_id"] == series_id]


def get_moment(hour):
    """
    return the moment of day
    night if 0 < hour < 6
    morning if 6 < hour < 12
    afternoon if 12 < hour < 18
    evening if 18 < hour < 24
    int -> str
    """
    if (0 < hour) and (hour < 6):
        return "night"
    elif (6 < hour) and (hour < 12):
        return "morning"
    elif (12 < hour) and (hour < 18):
        return "afternoon"
    else:
        return "evening"


# no return / will return fig?
def visualize_column(df, col_name, zero_line=False, std_line=False, mean_line=False):
    """
    plot col_name by step with eventually zero_line, std_line and mean_line, color by column awake.
    One plot by unique series_id in df
    df : DataFrame with columns col_name, series_id, step and awake
    DataFrame * str * bool * bool * bool
    """
    list_ids = df["series_id"].unique()

    plt.subplots(figsize=(20, len(list_ids) * 6))
    for i, id in enumerate(list_ids):
        plt.subplot(len(list_ids), 1, i+1)

        data_tmp = get_series(df, id)

        sns.lineplot(data=data_tmp, x="step", y=col_name, hue="awake", linewidth=0.5)

        if zero_line:
            plt.axhline(y = 0, color = 'r', linestyle = '-')
        if std_line:
            plt.axhline(y = data_tmp[col_name].std(), color = 'b', linestyle = '-', label="std")
        if mean_line:
            plt.axhline(y = data_tmp[col_name].mean(), color = 'g', linestyle = '-', label="mean")
        plt.title(f"Evolution of {col_name} - series {id}")
        plt.legend()
    plt.show()


def preprocess_col(df, col_name, rolling_val=1000):
    """
    return dataframe with same cols as df + rolling_mean, high_to_mean and centered for col_name

    col rolling_mean : mean value of col_name for window = rolling_val, by series_id
    high_to_mean column : values of rolling_mean or mean of rolling_mean by series_id if value > mean
    center : values of high_to_mean centered abs(on mean + std) / 2 by seies_id

    df: DataFrame with columns series_id, col_name

    DataFrame * str * int -> DataFrame
    """

    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:

        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # compute rolling_mean column
        df_tmp[f"{col_name}_rolling_mean"] =\
            df_tmp[col_name].rolling(window=rolling_val, center=True).mean().fillna(method="bfill").fillna(method="ffill")

        # compute high_to_mean column
        mean_value = df_tmp[f"{col_name}_rolling_mean"].mean()
        df_tmp[f"{col_name}_high_to_mean"] = df_tmp[f"{col_name}_rolling_mean"].apply(lambda x : mean_value if x > mean_value else x)

        # compute centered_column
        center_value = abs(df_tmp[f"{col_name}_high_to_mean"].mean() + df_tmp[f"{col_name}_high_to_mean"].std()) / 2
        df_tmp[f"{col_name}_centered"] = df_tmp[f"{col_name}_high_to_mean"].apply(lambda x : x - center_value)
        df_result = pd.concat([df_result, df_tmp])
    df_result = df_result.drop(columns=[f"{col_name}_rolling_mean", f"{col_name}_high_to_mean"])
    return df_result


def get_features(df):
    """
    Return DataFrame with more features from enmo and anglez:
    enmo_centered, datetime, hour, month, moment
    anglez_abs, anglez_diff, enmo_diff, anglez_rolling_mean, enmo_rolling_mean,
    enmo_x_anglez, enmo_x_anlez_abs, weekday, is_weekend
    DataFrame -> DataFrame
    """
    # mean number of rows by series
    size_series = df.shape[0] // df["series_id"].nunique()

    periods = 12
    # 12 * 5 secondes -> 1 minute

    # column enmo centered
    df_result = df.copy()

    # timestamp to datetime
    df_result["datetime"] = pd.to_datetime(df["timestamp"])

    # month
    df_result["month"] = df_result["datetime"].apply(lambda x : x.month)
    
    # hour
    df_result["hour"] = df_result["datetime"].apply(lambda x : x.hour)

    # moment of day
    df_result["moment"] = df_result["hour"].apply(lambda x : get_moment(x))
    
    # abs anglez
    df_result["anglez_abs"] = df["anglez"].apply(lambda x : abs(x))
    
    # diff between anglez_abs and n (periods) previous anglez
    # we need bfill because first values can't be computed
    df_result["anglez_diff"] = df_result.groupby("series_id")["anglez_abs"].diff(periods=periods).fillna(method="bfill")

    # diff between enmo and n (periods) previous enmo
    # we need bfill because first values can't be computed
    df_result["enmo_diff"] = df_result.groupby("series_id")["enmo"].diff(periods=periods).fillna(method="bfill")

    # rolling mean anglez abs
    # we need bfill and ffill because we have missing values at the begining and the end (center=True)
    df_result["anglez_rolling_mean"] = df_result["anglez_abs"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill")

    # rolling mean enmo
    # we need bfill and ffill because we have missing values at the begining and the end (center=True)
    df_result["enmo_rolling_mean"] = df["enmo"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill")
    
    
    # enmo * anglez
    df_result["enmo_x_anglez"] = df.apply(lambda x : x["enmo"] * x["anglez"], axis=1)
    
    # enmo * anglez_abs
    df_result["enmo_x_anglez_abs"] = df_result.apply(lambda x : x["enmo"] * x["anglez_abs"], axis=1)
    
    # is weekend
    df_result["weekday"] = df_result["datetime"].apply(lambda x : x.weekday())
    # Timestamp.weekday(): Monday == 0 … Sunday == 6.
    df_result["is_weekend"] = df_result["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    return df_result


def smooth_results(df, y_pred, smooth_val):
    """
    return a new array of predictions calculated with rolling mean for each series
    df : DataFrame with columns series_id
    y_pred : predictions for df (0 or )
    smooth_val : window of rolling
    DataFrame * array/Series * int -> DataFrame
    """
    list_ids = df["series_id"].unique()
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    y_result = pd.DataFrame(columns=y_pred.columns)

    for id in list_ids:
        # select series in df and y_pred
        df_tmp = get_series(df, id)
        start_id = df_tmp.index[0]
        end_id = df_tmp.index[-1]
        y_tmp = y_pred.iloc[start_id : end_id+1]
        
        y_tmp["pred"] =\
            y_tmp["pred"].rolling(window=smooth_val, center=True).mean().fillna(method="bfill").fillna(method="ffill")
        y_tmp["pred"] = y_tmp["pred"].apply(lambda x : 1 if x >= 0.5 else 0)
        
        y_result = pd.concat([y_result, y_tmp])


    return np.array(y_result["pred"]).astype("int")


def get_events(df, y_pred, y_probas):
    """
    Add column event, pred and score to df and return new df
    event = 0, 1 or np.nan
    
    df : DataFrame with columns series_id
    y_pred : DataFram with column pred of 0 and 1
    y_probas : dataFrame with probabilities from model
    
    DataFrame -> DataFrame
    """
    # add column pred and score to df
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    y_probas = pd.DataFrame(y_probas, columns=["score"])
    df = pd.concat([df, y_pred, y_probas], axis=1)
    
    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:
        
        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # create column diff
        df_tmp["pred_diff"] = df_tmp["pred"].diff().fillna(method="bfill")

        # use diff to determine event
        # when diff < 0 ie diff == -1 value went from 1 to 0 -> onset
        # when diff > 0 ie diff == 1 value went from 0 to 1 -> wakeup
        # 0 -> no changes
        df_tmp["event"] = df_tmp["pred_diff"].apply(lambda x : "onset" if x < 0 else ("wakeup" if x > 0 else np.nan))
        
        df_result = pd.concat([df_result, df_tmp])
        
    return df_result



def get_submission(df):
    """
    Returns a dataFrame ready for submission

    df : DataFrame with columns series_id, step, event, score
    """

    # remove nan values (no event)
    df = df.dropna()

    # keep only necessary columns
    df = df[["series_id", "step", "event", "score"]]

    # reset index
    df = df.reset_index(drop = True)

    # add first column: row_id
    row_id = df.index.values
    df.insert(0, 'row_id', row_id)

    return df


def check_couples(df):
    """
    check that first event of each series is "onset" and last event is "wakeup"
    else drop this row and print index of row
    reset index, row_id and return new dataframe
    df : dataFrame with column series_id and event
    """
    print("\nSTART")
    df_result = df
    list_ids = df_result["series_id"].unique()
    
    for id in list_ids:
        print(id)
        # get data frame for each series_id
        df_tmp = get_series(df_result, id)
        
        # get first and last event index
        start_id = df_tmp.index[0]
        end_id = df_tmp.index[-1]
    
        # check that first event is "onset" else drop row
        print(df.iloc[start_id, :]["event"])
        if df.iloc[start_id, :]["event"] != "onset":
            df_result = df_result.drop(start_id)
            print("!!!Removed row", start_id)
        
        # check that last event is "wakeup" else drop row
        print(df.iloc[end_id, :]["event"])
        if df.iloc[end_id, :]["event"] != "wakeup":
            df_result = df_result.drop(end_id)
            print("!!!Removed row", end_id)

    # reset index
    df_result = df_result.reset_index(drop = True)

    # reset row_id
    row_id = df_result.index.values
    df_result["row_id"] = row_id
 
    return df_result



def keep_periods(df, min_period):
    """
    Take a dataframe ready for submission and return same data frame minus periods of sleep < min_period
    Makes sure that sleep periods begin and end (start with onset, end with wakeup) for each series
    """
    
    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:

        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # get steps for onset and wakeup as lists
        pred_onsets = df_tmp[df_tmp["event"] == "onset"]["step"].to_list()
        pred_wakeups = df_tmp[df_tmp["event"] == "wakeup"]["step"].to_list()

        # check that all sleep periods start with onset and end wit wakeup
        # compare steps
        if min(pred_wakeups) < min(pred_onsets):     # first step of pred_wakeups smaller than first step of pred_onsets
            pred_wakeups = pred_wakeups[1:]          # don't keep first element of pred_wakeups
            print("delete wakeup")
        if max(pred_onsets) > max(pred_wakeups):     # last onset bigger than last wakeup
            pred_onsets = pred_onsets[:-1]           # don't keep last element of pred_onsets
            print("delete onset")


        # keep only sleep periods > min_period
        pred_onsets_2 = []
        pred_wakeups_2 = []
        for onset, wakeup in zip(pred_onsets, pred_wakeups):
            # we compare onset and wakeup couples
            if wakeup - onset >= min_period:
                pred_onsets_2.append(onset)
                pred_wakeups_2.append(wakeup)

        # keep only activity periods > min_period
        steps_to_keep = [pred_onsets_2[0]]               # keep first onset
        # we compare wakeup and onset couples
        for i, wakeup in enumerate(pred_wakeups_2[:-1]): # last wakeup can't be compared to any onset
            if pred_onsets_2[i+1] - wakeup >= min_period:
                steps_to_keep.append(wakeup)             # add couples of wakeup/onset
                steps_to_keep.append(pred_onsets_2[i+1])
        steps_to_keep.append(pred_wakeups_2[-1])        # add last wakeup event
        

        # select events en df_tmp
        df_tmp = df_tmp[df_tmp["step"].isin(steps_to_keep)]
        
        df_result = pd.concat([df_result, df_tmp])

        
    # reset index
    df_result = df_result.reset_index(drop = True)

    # reset row_id
    row_id = df_result.index.values
    df_result["row_id"] = row_id

    return df_result