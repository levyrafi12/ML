#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T, inverse_vocab):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n_samples = X_train.shape[0]
    D_t = [1 / n_samples for i in range(n_samples)]
    alpha_vals = []
    h_vals = []

    clf = [0] * n_samples

    for t in range(T):
        print("t = {}".format(t))
        h_pred, h_index, h_theta, eps_t = weak_learner_dt(D_t, X_train, y_train) # use decision tree

        print("{} {} {} {}".format(h_pred, inverse_vocab[h_index], h_theta, eps_t))

        h_vals.append((h_pred, h_index, h_theta))
        alpha_vals.append(0.5 * np.log((1 - eps_t) / eps_t)) # linear coefficient
        alpha_t = alpha_vals[-1]

        for i in range(n_samples):
            if X_train[i, int(h_index)] <= h_theta:
                clf[i] += alpha_t * h_pred
            else:
                clf[i] += alpha_t * -h_pred
        # print(clf)
        g_x = np.sign(clf)
        # print(g_x)
        print("train error {}".format(sum([int(y_train[i] != g_x[i]) for i in range(n_samples)])))

        Z_t = sum([D_t[i] * np.exp(-alpha_t * y_train[i] * hyp(h_pred, h_theta, X_train[i, int(h_index)])) \
            for i in range(n_samples)])
        D_t = [D_t[i] * np.exp(-alpha_t * y_train[i] * hyp(h_pred, h_theta, X_train[i, int(h_index)])) \
            / Z_t for i in range(n_samples)]

    return h_vals, alpha_vals

# You can add more methods here, if needed.

def weak_learner_dt(D, X_train, y_train):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    clf = DecisionTreeClassifier(random_state=0, max_depth=1)

    best_eps = 1

    for j in range(n_features):
        clf.fit(X_train[:, j].reshape(-1,1), y_train)
        y_pred = clf.predict(X_train[:, j].reshape(-1,1))
        theta = clf.tree_.threshold[0]
        # find the left leaf value
        if X_train[0, j] <= theta:
            sgn_val = y_pred[0] 
        else:
            sgn_val = -y_pred[0]

        eps = sum([int(y_train[i] != sgn_val) * D[i] if X_train[i, j] <= theta \
            else int(y_train[i] != -sgn_val) * D[i] for i in range(n_samples)])
        if eps < best_eps:
            best_theta = theta
            best_sgn_val = sgn_val
            best_ind = j
            best_eps = eps

    return best_sgn_val, best_ind, best_theta, best_eps

def weak_learner(D, X_train, y_train):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    best_eps = 1

    for j in range(n_features):
        word_counts = np.unique(X_train[:, j].reshape(-1,1)) 
        for theta in word_counts:
            for sgn_val in [-1, 1]:
                eps = sum([int(y_train[i] == -sgn_val) * D[i] if X_train[i, j] <= theta \
                    else int(y_train[i] == sgn_val) * D[i] for i in range(n_samples)])
                if eps < best_eps:
                    best_theta = theta
                    best_sgn_val = sgn_val
                    best_ind = j
                    best_eps = eps

    return best_sgn_val, best_ind, best_theta, best_eps

def hyp(h_pred, h_theta, word_count):
    if word_count <= h_theta:
        return h_pred
    return -h_pred

def training_error(h_vals, alpha_vals, X_train, y_train):
    T = [i for i in range(len(h_vals))]

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    err = []
    f = [0] * n_samples # f = sigma (alpha_t * h_t)

    for (h_pred, h_index, h_theta), alpha in zip(h_vals, alpha_vals):
        for i in range(n_samples):
            if X_train[i, int(h_index)] <= h_theta:
                y_val = h_pred 
            else:
                y_val = -h_pred
            f[i] += alpha_t * y_val

        g = np.sign(f)
        err.append(sum([int(y_train[i] != g[i]) for i in range(n_samples)]) / n_samples)

    plt.plot(T, err, 'b--')
    plt.xlabel('T')
    plt.ylabel('num errors')
    plt.text(len(T) / 2, 0.5, 'training error e_s(g)')
    plt.show()

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, inverse_vocab) = data

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    # for j in range(n_features):
    # print("{} {}".format(inverse_vocab[j], X_train[:, j]))

    h_vals, alpha_vals = run_adaboost(X_train, y_train, 80, inverse_vocab)

    # training_error(h_vals, alpha_vals, X_train, y_train)

    # for h_pred, h_index, h_theta in h_vals:
    # print("{} {} {}".format(h_pred, inverse_vocab[h_index], h_theta))

if __name__ == '__main__':
    main()



