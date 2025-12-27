import pandas as pd
import numpy as np
import os
import sys

def entropy(y):
    if len(y) == 0:
        return 0.0

    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def information_gain(feature, target, parent_entropy):
    total = len(target)

    left_target = target[feature > 0]
    right_target = target[feature <= 0]

    left_entropy = entropy(left_target)
    right_entropy = entropy(right_target)

    weighted_entropy = 0.0
    if len(left_target) > 0:
        weighted_entropy += (len(left_target) / total) * left_entropy
    if len(right_target) > 0:
        weighted_entropy += (len(right_target) / total) * right_entropy

    return parent_entropy - weighted_entropy


file = input()

data = pd.read_csv(os.path.join(sys.path[0], file))

fasting_blood = data["Fasting blood"].values
bmi = data["bmi"].values
family_history = data["FamilyHistory"].values
target = data["target"].values

parent_entropy = entropy(target)
print(f"Parent Node Entropy: {parent_entropy:.3f}")

ig_fasting = information_gain(fasting_blood, target, parent_entropy)
print(f"Information Gain (Fasting blood): {ig_fasting:.3f}")

ig_bmi = information_gain(bmi, target, parent_entropy)
print(f"Information Gain (bmi): {ig_bmi:.3f}")

ig_family = information_gain(family_history, target, parent_entropy)
print(f"Information Gain (FamilyHistory): {ig_family:.3f}")

ig_dict = {
    "Fasting blood": ig_fasting,
    "bmi": ig_bmi,
    "FamilyHistory": ig_family
}

best_feature = max(ig_dict, key=ig_dict.get)
best_ig = ig_dict[best_feature]

print(f"Best Feature for root node: {best_feature} with Information Gain: {best_ig:.3f}")
