import argparse
from utils import *
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import optimizers
from model import EncoderNet
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=4, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=500, help='feature dim')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='cora', help='name of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='number of clusters.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--threshold', type=float, default=0.5, help='the threshold of high-confidence')
parser.add_argument('--alpha', type=float, default=0.5, help='trade-off of loss')
args = parser.parse_args()

# Load data
adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

# Laplacian Smoothing
adj_norm_s = preprocess_graph(adj, args.t, norm='sym', renorm=True)
smooth_fea = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    smooth_fea = a.dot(smooth_fea)
smooth_fea = tf.constant(smooth_fea, dtype=tf.float32)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []

for seed in range(10):
    setup_seed(seed)

    # Init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis = clustering(smooth_fea.numpy(), true_labels,
                                                                           args.cluster_num)

    # MLP
    model = EncoderNet([features.shape[1], args.dims])
    optimizer = optimizers.Adam(learning_rate=args.lr)

    # GPU
    smooth_fea = tf.convert_to_tensor(smooth_fea.numpy(), dtype=tf.float32)
    sample_size = features.shape[0]
    target = tf.eye(smooth_fea.shape[0])

    for epoch in tqdm(range(args.epochs)):
        with tf.GradientTape() as tape:
            z1, z2 = model(smooth_fea, training=True)

            if epoch > 50:
                high_confidence = tf.reduce_min(dis, axis=1)
                threshold = tf.sort(high_confidence)[int(len(high_confidence) * args.threshold)].numpy()
                high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

                index = tf.range(smooth_fea.shape[0])[high_confidence_idx]
                y_sam = tf.convert_to_tensor(predict_labels, dtype=tf.int32)[high_confidence_idx]
                index = index[tf.argsort(y_sam)]
                class_num = {}

                for label in tf.sort(y_sam).numpy():
                    label = label.item()
                    if label in class_num.keys():
                        class_num[label] += 1
                    else:
                        class_num[label] = 1

                key = sorted(class_num.keys())
                if len(class_num) < 2:
                    continue

                pos_contrastive = 0
                centers_1 = tf.constant([], dtype=tf.float32)
                centers_2 = tf.constant([], dtype=tf.float32)

                for i in range(len(key[:-1])):
                    class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                    now = index[class_num[key[i]]:class_num[key[i + 1]]]
                    pos_embed_1 = tf.gather(z1, np.random.choice(now.numpy(), size=int((now.shape[0] * 0.8)),
                                                                   replace=False))
                    pos_embed_2 = tf.gather(z2, np.random.choice(now.numpy(), size=int((now.shape[0] * 0.8)),
                                                                   replace=False))
                    pos_contrastive += tf.reduce_sum(2 - 2 * pos_embed_1 * pos_embed_2)

                    centers_1 = tf.concat([centers_1, tf.reduce_mean(z1[now], axis=0, keepdims=True)], axis=0)
                    centers_2 = tf.concat([centers_2, tf.reduce_mean(z2[now], axis=0, keepdims=True)], axis=0)

                pos_contrastive = pos_contrastive / args.cluster_num

                if pos_contrastive == 0:
                    continue

                if len(class_num) < 2:
                    loss = pos_contrastive
                else:
                    centers_1 = tf.nn.l2_normalize(centers_1, axis=1)
                    centers_2 = tf.nn.l2_normalize(centers_2, axis=1)
                    S = tf.matmul(centers_1, centers_2, transpose_b=True)
                    S_diag = tf.linalg.diag_part(S)
                    S = S - tf.linalg.diag(S_diag)
                    neg_contrastive = tf.reduce_mean(tf.square(S))
                    loss = pos_contrastive + args.alpha * neg_contrastive

            else:
                S = tf.matmul(z1, z2, transpose_b=True)
                loss = tf.losses.mean_squared_error(S, target)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 10 == 0:
            z1, z2 = model(smooth_fea, training=False)

            hidden_emb = (z1 + z2) / 2

            acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb.numpy(), true_labels, args.cluster_num)
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean(), "±", acc_list.std())
print(nmi_list.mean(), "±", nmi_list.std())
print(ari_list.mean(), "±", ari_list.std())
print(f1_list.mean(), "±", f1_list.std())
