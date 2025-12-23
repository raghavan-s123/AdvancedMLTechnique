def bays_theorem(p_disease, p_no_disease, p_positive_given_dis, p_pos_given_no_dis):
    
    p_positive = (p_positive_given_dis * p_disease) + (p_pos_given_no_dis * p_no_disease)
    prob = (p_positive_given_dis * p_disease) / p_positive
    print(f"Probability of having disease given a positive test: {prob:.4f}")