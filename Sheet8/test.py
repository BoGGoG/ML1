import pandas as pd
import numpy as np
import sys
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

testt_file = 'only_testt.dat'
testt = pd.read_csv(testt_file, sep = '\t', header = None)

predy = pd.read_csv(sys.stdin, sep = '\t', header = None)

# for i in range(0,20):
    # print(predy.values[i], testt.values[i])

testt_arr = testt.values
predy_arr = predy.values

# >0: FN
# <0: FP
diff_arr = testt_arr - predy_arr
fp_arr = diff_arr[diff_arr < 0]
fn_arr = diff_arr[diff_arr > 0]
tp_arr = predy_arr[testt_arr > 0]
tp_arr = tp_arr[tp_arr > 0]
tn_arr = predy_arr[testt_arr == 0]
tn_arr = tn_arr[tn_arr == 0]
correct_arr = diff_arr[diff_arr == 0]

print('Test data number:', diff_arr.shape)
print('False positives:', fp_arr.shape[0])
print('False negatives', fn_arr.shape[0])
print('True positives', tp_arr.shape[0])
print('True negatives', tn_arr.shape[0])
print('Correct: ', correct_arr.shape[0] / diff_arr.shape[0])

# calculate loss
# FN + 5 FP
# loss = (np.sum(np.abs(fn_arr)) +  5. * np.sum(np.abs(fp_arr))) / diff_arr.shape[0]
# print("loss: (FN + 5 * FP) / All ", loss)
loss = (5. * np.sum(np.abs(fn_arr)) +  np.sum(np.abs(fp_arr))) / diff_arr.shape[0]
print("loss: (5 * FN + FP) / All ", loss)

print("1 - loss: ", 1 - loss)
print("Points: (1 - loss) * 20 =  ", (1 - loss) * 20)
print(classification_report(testt, predy))
