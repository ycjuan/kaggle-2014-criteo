#!/usr/bin/env python

import os, sys

usage_string = 'usage: txt2csv.py {tr|te} input output'

if len(sys.argv) != 4:
    print(usage_string)
    exit(1)

type, src_path, dst_path = sys.argv[1:]

if type == 'tr':
    header = 'Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26' 
    idx = 10000000
elif type == 'te':
    header = 'Id,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26' 
    idx = 60000000
else:
    print(usage_string)
    exit(1)

with open(dst_path, 'w') as f:
    f.write(header + '\r\n')
    for line in open(src_path, 'r'):
        line = str(idx) + ',' + line.replace('\t', ',')
        f.write(line.replace('\n', '\r\n'))
        idx += 1
