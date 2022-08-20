from utils.WuEnda.public_tests import *
import matplotlib.pyplot as plt
import numpy as np


def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    temp_idx = 0
    for i in range(X.shape[0]):
        temp_distance = np.finfo(np.float32).max
        for j in range(centroids.shape[0]):
            distance = np.linalg.norm(X[i] - centroids[j])
            if distance < temp_distance:
                temp_distance = distance
                temp_idx = j

        idx[i] = temp_idx

    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    group_count = np.zeros(K)

    for i in range(m):
        centroids[idx[i]] += X[i]
        group_count[idx[i]] += 1

    for i in range(K):
        centroids[i] /= group_count[i]

    return centroids


def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    def draw_line(p1, p2, style="-k", linewidth=1):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

    # Plot the examples
    plt.scatter(X[:, 0], X[:, 1], idx)
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    plt.title("Iteration number %d" % i)


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx


def k_means_for_numpy_data(K=3, max_iters=10, random_init=False):
    # numpy data
    X = np.load("./k_means/data/ex7_X.npy")
    if random_init:
        initial_centroids = kMeans_init_centroids(X, K)
    else:
        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)


def k_means_for_image_compress(K=16, max_iters=10):
    # Load an image of a bird
    original_img = plt.imread('./k_means/data/bird_small.png')
    # Visualizing the image
    plt.imshow(original_img)
    print("Shape of original_img is:", original_img.shape)

    # Divide by 255 so that all values are in the range 0 - 1
    original_img = original_img / 255

    # Reshape the image into an m x 3 matrix where m = number of pixels
    # (in this case m = 128 x 128 = 16384)
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X_img that we will use K-Means on.
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

    # Using the function you have implemented above.
    initial_centroids = kMeans_init_centroids(X_img, K)
    # Run K-Means - this takes a couple of minutes
    centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
    print("Shape of idx:", idx.shape)
    print("Closest centroid for the first five elements:", idx[:5])
    # Represent image in terms of indices
    X_recovered = centroids[idx, :]
    # Reshape recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, original_img.shape)
    # Display original image
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    plt.axis('off')

    ax[0].imshow(original_img * 255)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    # Display compressed image
    ax[1].imshow(X_recovered * 255)
    ax[1].set_title('Compressed with %d colours' % K)
    ax[1].set_axis_off()
    plt.show()


def k_means(test_api=False):
    if test_api:
        X = np.load("data/ex7_X.npy")
        print("First five elements of X are:\n", X[:5])
        print('The shape of X is:', X.shape)

        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
        idx = find_closest_centroids(X, initial_centroids)
        print("First three elements in idx are:", idx[:3])
        find_closest_centroids_test(find_closest_centroids)

        test_centroids = compute_centroids(X, idx, 3)
        print("The centroids are:", test_centroids)
        compute_centroids_test(compute_centroids)

    k_means_for_numpy_data(random_init=True)
    k_means_for_image_compress()
