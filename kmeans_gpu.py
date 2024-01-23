import numpy as np
import tensorflow as tf

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (tf.Tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = X.shape[0]
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = tf.gather(X, indices)
    return initial_state.numpy()

def kmeans(X, num_clusters, distance='euclidean', tol=1e-4):
    """
    perform kmeans
    :param X: (tf.Tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :return: (tf.Tensor, tf.Tensor) cluster ids, cluster centers
    """
    def pairwise_distance(data1, data2):
        return tf.reduce_sum(tf.square(data1[:, tf.newaxis, :] - data2), axis=-1)

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = tf.losses.cosine_distance
    else:
        raise NotImplementedError

    # convert to float32
    X = tf.cast(X, dtype=tf.float32)

    # initialize
    dis_min = float('inf')
    initial_state_best = None
    for _ in range(20):
        initial_state = initialize(X, num_clusters)
        dis = tf.reduce_sum(pairwise_distance_function(X, initial_state))
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = tf.argmin(dis, axis=1)

        initial_state_pre = tf.identity(initial_state)

        for index in range(num_clusters):
            selected = tf.where(tf.equal(choice_cluster, index))[:, 0]
            selected = tf.gather(X, selected)
            initial_state[index] = tf.reduce_mean(selected, axis=0)

        center_shift = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(initial_state - initial_state_pre), axis=1)))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster, dis, initial_state

def kmeans_predict(X, cluster_centers, distance='euclidean'):
    """
    predict using cluster centers
    :param X: (tf.Tensor) matrix
    :param cluster_centers: (tf.Tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :return: (tf.Tensor) cluster ids
    """
    def pairwise_distance(data1, data2):
        return tf.reduce_sum(tf.square(data1[:, tf.newaxis, :] - data2), axis=-1)

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = tf.losses.cosine_distance
    else:
        raise NotImplementedError

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = tf.argmin(dis, axis=1)

    return choice_cluster.numpy()
