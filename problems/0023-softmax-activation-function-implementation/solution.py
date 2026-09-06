import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    sum_exp = sum(exp_scores)
    result = [score/sum_exp for score in exp_scores]
    return result