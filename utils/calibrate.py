#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('src_path', type=str)
parser.add_argument('dst_path', type=str)
args = vars(parser.parse_args())

with open(args['dst_path'], 'w') as f:
    for line in open(args['src_path']):
        prediction = float(line.strip())
        if prediction > 0.0035:
            prediction -= 0.003
        f.write('{0}\n'.format(prediction))
