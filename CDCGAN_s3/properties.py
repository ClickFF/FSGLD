import numpy as np
from sklearn.metrics import pairwise_distances

def calc_metrics(generated, real):
    """计算所有评估指标"""
    gen_round = np.round(generated)
    real_round = np.round(real)
    
    # 唯一性
    unique = len(set(tuple(row) for row in gen_round))
    
    # 新颖性
    real_set = set(tuple(row) for row in real_round)
    novel = sum(1 for row in gen_round if tuple(row) not in real_set)
    
    # 多样性
    jaccard_dist = pairwise_distances(gen_round, metric='jaccard')
    
    # 相似性
    jaccard_sim = 1 - pairwise_distances(gen_round, real_round, metric='jaccard')
    
    return {
        'uniqueness': unique / len(gen_round),
        'novelty': novel / len(gen_round),
        'diversity': jaccard_dist.mean(),
        'avg_similarity': jaccard_sim.mean(),
        'max_similarity': jaccard_sim.max()
    }