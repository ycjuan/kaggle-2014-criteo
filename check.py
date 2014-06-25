#!/usr/bin/env python3

import subprocess, os

def check(path, target_md5sum):
    if not os.path.exists(path):
        print('{0} does not exist'.format(path))
        return False
    p = subprocess.Popen('md5sum {0}'.format(path).split(), stdout=subprocess.PIPE)
    stdout = p.stdout.readline().decode('utf-8')
    md5sum = stdout.split()[0]
    if md5sum == target_md5sum:
        return True
    else:
        print('{0} is incorrect'.format(path))
        return False

success = True
if not check('train.csv', 'ebf87fe3daa7d729e5c9302947050f41'):
    success = False 
if not check('test.csv', '8016f59e45abb37ae7f6e7956f30e052'):
    success = False 

if success:
    print('Your csv files are correct!')
