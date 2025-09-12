import numpy as np
from sklearn.metrics import pairwise_distances

def calc_metrics(generated, real):
    """计算所有评估指标"""
    gen_round = np.round(generated)
    real_round = np.round(real)
    
    # uniqueness 
    unique = len(set(tuple(row) for row in gen_round))
    
    # novelty
    real_set = set(tuple(row) for row in real_round)
    novel = sum(1 for row in gen_round if tuple(row) not in real_set)
    
    # diversity
    jaccard_dist = pairwise_distances(gen_round, metric='jaccard')
    
    # similarity
    jaccard_sim = 1 - pairwise_distances(gen_round, real_round, metric='jaccard')
    
    return {
        'uniqueness': unique / len(gen_round),
        'novelty': novel / len(gen_round),
        'diversity': jaccard_dist.mean(),
        'avg_similarity': jaccard_sim.mean(),
        'max_similarity': jaccard_sim.max()
    }
