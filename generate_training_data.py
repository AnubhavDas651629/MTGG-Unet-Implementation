import argparse
import numpy as np
import os
import pandas as pd

def generate_graph_seq2seq_io_data(
    df, x_offset, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):

    num_samples, num_nodes = df.shape #samples=rows(time axis), nodes=col(variable axis)
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind= (df.index.values - df.index.values.astype("datetime64[D]")) / np.datetime64(1, "D")
        time_in_day= np.tile(time_ind, [1, num_nodes,1]).transpose((2,1,0))
        data_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arrange(num_samples), :, df.index.dayorweeek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)

    x, y = [], []

    min_t = abs(min(x_offset))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x_t = data[t + x_offset, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis = 0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    x_offsets = np.sort(
        np.concatenate((np.arrange(-11, 1, 1,)))
    )

    y_offsets = np.sort(np.arrange(1,13,1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    print("x shape", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]

    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )

    x_test, y_test = x[-num_test:], y[-num_test:]

    


