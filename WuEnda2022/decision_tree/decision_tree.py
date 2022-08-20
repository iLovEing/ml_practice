from utils.WuEnda.public_tests import *
import numpy as np


def compute_entropy(y):
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    entropy = 0.

    if len(y) != 0:
        sum = y.sum()

        if sum != len(y) and sum != 0:
            p1 = sum / len(y)
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    return entropy


def split_dataset(X, node_indices, feature):
    """
        Splits the data at the given node into
        left and right branches

        Args:
            X (ndarray):             Data matrix of shape(n_samples, n_features)
            node_indices (ndarray):  List containing the active indices. I.e, the samples being considered at this step.
            feature (int):           Index of feature to split on

        Returns:
            left_indices (ndarray): Indices with feature value == 1
            right_indices (ndarray): Indices with feature value == 0
    """
    left_indices = []
    right_indices = []

    for it in node_indices:
        if X[it][feature] == 1:
            left_indices.append(it)
        else:
            right_indices.append(it)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    information_gain = 0.
    if len(node_indices) != 0:
        left_indices, right_indices = split_dataset(X, node_indices, feature)

        # Some useful variables
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        w_left = len(left_indices) / len(node_indices)
        information_gain = compute_entropy(y_node) - (w_left * compute_entropy(y_left) +
                                                      (1 - w_left) * compute_entropy(y_right))

    return information_gain


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    num_features = X.shape[1]

    best_feature = -1
    temp_gain = 0.
    for it in range(num_features):
        gain = compute_information_gain(X, y, node_indices, it)
        if gain > temp_gain:
            temp_gain = gain
            best_feature = it

    return best_feature


def build_tree_recursive(tree, X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree.
        current_depth (int):    Current depth. Parameter used during recursive call.

    """

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)
    tree.append((current_depth, branch_name, best_feature, node_indices))

    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(tree, X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(tree, X, y, right_indices, "Right", max_depth, current_depth + 1)


def decision_tree(test_api=False):
    # 生成数据
    X_train = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 1, 1],
                        [0, 1, 1],
                        [0, 0, 0],
                        [1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    print("First few elements of X_train:\n", X_train[:5])
    print("Type of X_train:", type(X_train))

    print("First few elements of y_train:", y_train[:5])
    print("Type of y_train:", type(y_train))

    print('The shape of X_train is:', X_train.shape)
    print('The shape of y_train is: ', y_train.shape)
    print('Number of training examples (m):', len(X_train))

    if test_api:
        # Compute entropy at the root node (i.e. with all examples)
        # Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"
        print("Entropy at root node: ", compute_entropy(y_train))
        compute_entropy_test(compute_entropy)

        root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Feel free to play around with these variables
        # The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
        feature = 0
        left_indices, right_indices = split_dataset(X_train, root_indices, feature)
        print("Left indices: ", left_indices)
        print("Right indices: ", right_indices)
        split_dataset_test(split_dataset)

        info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
        print("Information Gain from splitting the root on brown cap: ", info_gain0)
        info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
        print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)
        info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
        print("Information Gain from splitting the root on solitary: ", info_gain2)
        compute_information_gain_test(compute_information_gain)

        best_feature = get_best_split(X_train, y_train, root_indices)
        print("Best feature to split on: %d" % best_feature)
        get_best_split_test(get_best_split)

    tree = []
    build_tree_recursive(tree, X_train, y_train, range(X_train.shape[0]), "Root", max_depth=3, current_depth=0)
    print(tree)