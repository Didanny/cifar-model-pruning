import csv
import argparse

import torch

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt