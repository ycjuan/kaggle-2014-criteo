#!/usr/bin/env python3

import argparse, sys

from common import *

def parse_args():
    
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='nr_thread', default=12, type=int)
    parser.add_argument('cvt_path')
    parser.add_argument('src1_path')
    parser.add_argument('src2_path')
    parser.add_argument('dst_path')
    args = vars(parser.parse_args())

    return args

def main():
    
    args = parse_args()

    nr_thread = args['nr_thread']
    
    split(args['src1_path'], nr_thread, True)

    split(args['src2_path'], nr_thread, False)

    parallel_convert(args['cvt_path'], [args['src1_path'], args['src2_path'], args['dst_path']], nr_thread)

    cat(args['dst_path'], nr_thread)

    delete(args['src1_path'], nr_thread)

    delete(args['src2_path'], nr_thread)

    delete(args['dst_path'], nr_thread)

main()
