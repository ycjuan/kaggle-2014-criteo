#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('out_path', type=str)
parser.add_argument('submission_path', type=str)
args = parser.parse_args()

OUT_PATH, SUB_PATH = args.out_path, args.submission_path

i = 60000000
with open(SUB_PATH, 'w') as f:
    f.write('Id,Predicted\n')
    for i, line in enumerate(open(OUT_PATH)):
        f.write('{0},{1}'.format(i+60000000, line))
