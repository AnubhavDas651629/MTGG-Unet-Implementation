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