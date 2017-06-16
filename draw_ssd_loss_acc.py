# -*- coding: utf-8 -*-
"""
* @File Name:   		draw_ssd_loss_acc.py
* @Author:				Wang Yang
* @Created Date:		2017-06-16 16:52:04
* @Last Modified Data:	2017-06-16 17:47:58
* @Desc:					
*
"""

import sys
import re
import matplotlib.pyplot as plt
import numpy as np

# print('argv is %s' % sys.argv)

if len(sys.argv) != 2:
	print('usage: python draw_ssd_loss_acc.py log_file')
	sys.exit()

log_file = sys.argv[1]

fp = open(log_file, 'r+')

text = fp.read()

all_res = re.findall(r"Iteration (\w+).*\n.*mbox_loss = ([\w\.]+)", text)

len_of_res = len(all_res)

mbox_loss_vec = np.arange(2*len_of_res, dtype='float32').reshape(len_of_res, 2)

for i in range(len_of_res):
	res = all_res[i]

	iter_no = int(res[0])
	loss = float(res[1])
	mbox_loss_vec[i][0] = iter_no
	mbox_loss_vec[i][1] = loss

	pass

# print(mbox_loss_vec)

all_res = re.findall(r"Iteration (\w+), Testing net[^=]+detection_eval = ([\w\.]+)", text)

len_of_res = len(all_res)
eval_val_vec = np.arange(2*len_of_res, dtype='float32').reshape(len_of_res, 2)

for i in range(len_of_res):
	res = all_res[i]

	iter_no = int(res[0])
	eval_val = float(res[1])

	eval_val_vec[i][0] = iter_no
	eval_val_vec[i][1] = eval_val

	pass

# print(eval_val_vec)

plt.plot(mbox_loss_vec[:, 0], mbox_loss_vec[:, 1], label='loss')
plt.plot(eval_val_vec[:, 0], eval_val_vec[:, 1], label='eval')

plt.legend()

plt.show()

