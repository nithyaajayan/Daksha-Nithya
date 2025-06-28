# plot.py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import plotdistribution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1_file",type=str,required=True,help=".npy file to dataset1")
    parser.add_argument("--data2_file",type=str,required=True,help=".npy file to dataset2")
    parser.add_argument("--true_value",type=float,required=True)
    parser.add_argument("--data1_label",type=str,required=True)
    parser.add_argument("--data2_label",type=str,required=True)
    parser.add_argument("--title",type=str,required=True)
    parser.add_argument("--xlabel",type=str,required=True)
    parser.add_argument("--bins",type=int)
    args = parser.parse_args()
    
    data1 = np.load(args.data1_file)
    data2 = np.load(args.data2_file)
    
    bins = args.bins if args.bins else int(np.sqrt(data1.shape[0]))
    
    plotdistribution(data1,data2,args.true_value,args.data1_label,args.data2_label,
                     args.title, args.xlabel,bins)
