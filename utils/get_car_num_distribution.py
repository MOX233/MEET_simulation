from typing import DefaultDict
from utils.read_tripInfo import read_tripInfo
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser

args = args_parser()
car_tripinfo = read_tripInfo(tripInfo_path='tripinfo.xml')
FL_table = generate_FLtable_from_tripInfo(args)

import ipdb;ipdb.set_trace()

carnum_list = []
for k,v in FL_table.items():
    carnum_list.append(len(v.keys()))

carnum_list.append(0)
print("每轮参与训练的小车数量之和: ", sum(carnum_list))
import matplotlib.pyplot as plt
plt.figure(1)
plt.hist(carnum_list,bins=max(carnum_list)+1,range=(0,max(carnum_list)+1), density=False)
plt.savefig('test.png')
