import numpy as np
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_ms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors

def hadamard_op_comm(arg1, arg2, arg3):
    '''
    Generate features for a Node-pair using their vector representation and community information.
    '''
    feat = np.multiply(arg1, arg2).tolist()  # Use numpy for element-wise multiplication
    feat.append(int(arg3))
    return feat

def edge_embedding_comm(node_embedding, edges):
    '''
    Call hadamard_op_comm function to generate feature representation for all node pairs.
    '''
    edge_embs = []
    for (n1, n2, n3) in edges:
        n1_emb = node_embedding[n1]
        n2_emb = node_embedding[n2]
        edge_embs.append(hadamard_op_comm(n1_emb, n2_emb, n3))
    return edge_embs

def load_data_comm(filename, chunk_size=1000):
    '''
    Read train and test data in chunks
    '''
    with open(filename, 'r') as f:
        edges = []
        for line in f:
            tmp = line.strip().split()
            edges.append((tmp[0], tmp[1], tmp[2]))
            if len(edges) == chunk_size:
                yield edges
                edges = []
        if edges:
            yield edges

def link_prediction():
    emb = KeyedVectors.load_word2vec_format("Output/sample.emb", binary=False)

    log_reg = sk_lm.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)

    # Load train data in chunks
    for train_edges in load_data_comm("Input/sample_trainx.txt"):
        x_train = edge_embedding_comm(emb, train_edges)
        y_train = np.loadtxt("Input/sample_trainy.txt", max_rows=len(train_edges))

        # Convert to numpy array to save memory
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        log_reg.partial_fit(x_train, y_train, classes=np.array([0, 1]))

    # Load test data
    x_test = []
    y_test = []
    for test_edges in load_data_comm("Input/sample_test.txt"):
        x_test.extend(edge_embedding_comm(emb, test_edges))
        y_test.extend(np.loadtxt("Input/sample_testy.txt", max_rows=len(test_edges)))

    # Convert to numpy array to save memory
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_pred = log_reg.predict(x_test)

    # Check accuracy
    accuracy = sk_ms.accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('Total Accuracy: ', accuracy)
    print('Total AUC: ', roc_auc)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)

    # Save predictions to a file with headers and descriptive labels
    with open("Output/predictions.txt", "w") as f:
        f.write("Predicted Label\tTrue Label\n")
        for true_label, pred_label in zip(y_test, y_pred):
            true_label_str = "TRUE" if true_label == 1 else "FALSE"
            pred_label_str = "TRUE" if pred_label == 1 else "FALSE"
            f.write(f"{pred_label_str}\t{true_label_str}\n")

link_prediction()
