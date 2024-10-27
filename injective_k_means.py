def injective_k_means(vectors, districts_per_plan, k, maxsteps=10000, verbose=True, seed=2024):
    '''
    Assumes the vectors are ordered by plan (i.e. the first districts_per_plan vectors are the first plan)

    vectors is a (number of districts)x(number of units) numpy array
    districts_per_plan is the number of districts in each plan, e.g. 14 in NC Congressional
    k is the number of clusters desired, must be more than districts_per_plan

    Returns the cluster centroids, the labels for the vectors, and the inertia to use in an elbow plot
    '''
    
    assert k >= districts_per_plan
    np.random.seed(seed)
    initial = np.random.choice(range(len(vectors)), size=k, replace=False)
    means = np.array([vectors[i] for i in initial])
    labels = np.zeros(len(vs))
    oldlabels = np.zeros(len(vs))
    
    for step in range(maxsteps):
        #match
        if verbose:
            print('.', end='')
        M = distance_matrix(vectors, means)
        #first just greedy match
        greedy_matches = np.argmin(M, axis=1)
        labels = greedy_matches

        adjust = 0
        for i in range(0, len(vectors), districts_per_plan):
            if len(set(labels[i:i+districts_per_plan])) < districts_per_plan:
                rows, cols = LSA(M[i:i+districts_per_plan,:])
                labels[i:i+districts_per_plan] = cols
                adjust += 1
        if verbose:
            print('{} adjusted'.format(adjust))

        #check convergence
        if np.array_equal(oldlabels, labels):
            print('CONVERGED in {} steps'.format(step))
            break
        else:
            oldlabels = labels.copy()
        if step == maxsteps-1:
            print('WARNING: DID NOT CONVERGE!')

        #average
        if verbose:
            print(':', end='')
        means = np.array([
            np.mean([v for v,m in zip(vectors, labels) if m == i], axis=0) for i in range(k)
        ])

    inertia = np.sum([M[i, labels[i]]**2 for i in range(len(vectors))])
    return means, labels, inertia

