import os
import sys
import pandas as pd
from ML_Modules import bays_theorem

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

p_disease = 0.001
p_no_disease = 1 - p_disease
p_positive_given_dis = 0.95

print("Scenario 1: False positive rate = 5%")
bays_theorem(p_disease, p_no_disease, p_positive_given_dis, 0.05)
print()

print("Scenario 2: False positive rate = 10%")
bays_theorem(p_disease, p_no_disease, p_positive_given_dis, 0.10)




