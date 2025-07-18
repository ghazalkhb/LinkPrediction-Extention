import argparse
import networkx as nx
import nodesim
from gensim.models import Word2Vec
import time


def parse_args():
    '''
    This function parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run NodeSim.")

    parser.add_argument('--input', nargs='?', default='Input/new/train_nodes_edges.txt',
                        help='Path of input graph')

    parser.add_argument('--output', nargs='?', default='Output/samplen.emb',
                        help='Path of output embeddings')

    parser.add_argument('--input-format', nargs='?', default='custom',
                        help='Format of input graph file (e.g., custom). Default is custom.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per node. Default is 10.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size or window size. Default is 5.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--a', type=float, default=1,
                        help='This is alpha parameter for NodeSim Method that is intra community prob-weight. Default is 1.')

    parser.add_argument('--b', type=float, default=2,
                        help='This is beta parameter for NodeSim Method that is inter community prob-weight. Default is 1.5.')

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network from a custom formatted text file.
    '''
    G = nx.Graph()
    with open(args.input, 'r') as file:
        lines = file.readlines()

        # Read nodes
        node_section = True
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            if line == "Edges:":
                node_section = False
                continue

            if node_section:
                if line.startswith("Nodes:"):
                    continue
                parts = line.split(": ", 1)  # Split only at the first occurrence of ": "
                if len(parts) != 2:
                    raise ValueError(f"Line format is incorrect for node: {line}")
                node_id, attrs = parts
                attrs = eval(attrs)  # Convert string dict to actual dict
                G.add_node(node_id, **attrs)
            else:
                edge = eval(line)
                G.add_edge(*edge[:2], **edge[2])

    return G


def learn_embeddings(walks):
    '''
    Learn network embeddings using Skipgram model.
    '''
    # Run the following 6 lines if the node type is not string
    new_walks = []
    for walk in walks:
        w1 = []
        for i in walk:
            w1.append(str(i))
        new_walks.append(w1)
    print("in embedding")
    model = Word2Vec(sentences=new_walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
    model.wv.save_word2vec_format(args.output)
    return


def main(args):
    '''
    Method executes following functions: read network, execute nodesim random walks, and learn embedding.
    '''
    nx_G = read_graph()
    G = nodesim.Graph(nx_G, args.a, args.b)
    G.compute_edge_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    starttime = time.time()
    main(args)
    endtime = time.time()
    print(endtime - starttime)
