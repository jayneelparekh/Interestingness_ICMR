import numpy as np


def mih_to(PPM):
    # Shape of PPM (Pairwise Prediction Matrix) should be square
    n_elem = np.shape(PPM)[0]

    if (n_elem == 1):
	return np.array([1])

    else:
	others_PPM = PPM[1:, 1:]
	rank_others = mih_to(others_PPM)
	best_inv = 200
	best_i = 0
	for i in range(1, n_elem+1):
	    vec = PPM[0, :]
	    extra_inv = np.sum(np.logical_and(rank_others >= i, vec[1:] == 0)) + np.sum(np.logical_and(rank_others < i, vec[1:] == 1))
	    if (extra_inv < best_inv):
		best_i = i
	    best_inv = min(best_inv, extra_inv)

	rank_others = rank_others + (rank_others >= best_i).astype(int)
	new_rank = np.array(range(n_elem))
	new_rank[0] = best_i
	new_rank[1:] = rank_others
	return new_rank





def mih_ro(PPM):
    # Shape of PPM (Pairwise Prediction Matrix) should be square
    n_elem = np.shape(PPM)[0]

    if (n_elem == 1):
	return np.array([1])

    else:
	idx = np.random.randint(0, n_elem)
	others_PPM = np.delete(PPM, idx, axis=0)
	others_PPM = np.delete(others_PPM, idx, axis=1)
	rank_others = mih_ro(others_PPM)
	best_inv = 200
	best_i = 0
	for i in range(1, n_elem+1):
	    vec = np.append((PPM[0:idx, idx] == 0).astype(int), PPM[idx, idx+1:])
	    extra_inv = np.sum(np.logical_and(rank_others >= i, vec == 0)) + np.sum(np.logical_and(rank_others < i, vec == 1))
	    if (extra_inv < best_inv):
		best_i = i
	    best_inv = min(best_inv, extra_inv)

	rank_others = rank_others + (rank_others >= best_i).astype(int)
	new_rank = np.array(range(n_elem))
	new_rank[idx] = best_i
	new_rank[0:idx] = rank_others[0:idx]
	new_rank[idx+1:] = rank_others[idx:]
	return new_rank



