# -*- coding: utf-8 -*-

import os

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import argparse


def _to_csv(export_path, save_data, columns, index=False):
    df = pd.DataFrame(data=save_data, columns=columns)
    df.to_csv(export_path, index=index)


def single_event_scalars_export_tool(in_path, ex_path):
    event_data = event_accumulator.EventAccumulator(in_path)
    event_data.Reload()

    keys = event_data.scalars.Keys()
    print("keys list: {}".format(keys))

    export_items = ["step", "wall_time", "value"]
    for key in keys:
        print("process key: {}".format(key))
        export_names = ["{}_{}".format(key, item) for item in export_items]
        save_data = list()
        for e in event_data.Scalars(key):
            temp_data = [e.step, e.wall_time, e.value]
            save_data.append(temp_data)

        save_name = "{}.csv".format(key.replace("/", "_"))
        save_path = os.path.join(ex_path, save_name)
        _to_csv(save_path, save_data, export_names)
        print("export data of {} done...".format(key))


def multi_event_scalars_export_tool(in_path, ex_path):
    root, dirs, _ = next(os.walk(in_path))
    for _dir in dirs:
        print("process summary: {}".format(_dir))
        event_in_path = os.path.join(root, _dir, "tensorboard_log")
        event_ex_path = os.path.join(ex_path, _dir)
        if not os.path.exists(event_ex_path):
            os.makedirs(event_ex_path)
        single_event_scalars_export_tool(event_in_path, event_ex_path)


def single_event_images_export_tool():
    pass


def multi_event_images_export_tool():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export tensorboard data")
    parser.add_argument("--in_path", type=str, help="tensorboard file location")
    parser.add_argument("--ex_path", type=str, default="./", help="export path")
    parser.add_argument("--state", type=str, default="single", help="single summary or multi summaries")
    parser.add_argument("--ex_type", type=str, default="scales", help="export data type")
    args = parser.parse_args()

    if args.state == "single":
        if args.ex_type == "scales":
            single_event_scalars_export_tool(args.in_path, args.ex_path)
        else:
            raise KeyError
    elif args.state == "multi":
        if args.ex_type == "scales":
            multi_event_scalars_export_tool(args.in_path, args.ex_path)
        else:
            raise KeyError
