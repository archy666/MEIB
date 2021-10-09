"""
This demo is for "MULTI VIEW INFORMATION BOTTLENECK WITHOUT VARIATIONAL APPROXIMATION" in ICASSP 2022.
"""


import numpy as np
import scipy.io as sio
import torch
from model import MLP
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
from scipy.spatial.distance import pdist, squareform
from utils import calculate_MI
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='MEIB')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', '-wd', type=float, default=0.03, help='weight_decay. Default:5e-4')
parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs. Default: 10')
parser.add_argument('--batchsize', type=int, default=64, help='batch_size')
parser.add_argument('--ks', type=int, default=10, help='kernel size')
args = parser.parse_args()

# ---------------------------------------------- Load data function ---------------------------------------------------
def load_data(validation, iterr):
    data = sio.loadmat('synthetic_data/iter' + str(iterr) + '.mat')
    Xtrain1 = data['X1_train']
    Xtrain2 = data['X2_train']
    ytrain = data['ytrain']
    if validation:
        Xtest1 = data['X1_val']
        Xtest2 = data['X2_val']
        ytest = data['yval']
    else:
        Xtest1 = data['X1_test']
        Xtest2 = data['X2_test']
        ytest = data['ytest']
    ytrain = np.squeeze(ytrain)
    ytest = np.squeeze(ytest)
    num_examples, _ = Xtrain1.shape
    idx = np.random.permutation(num_examples)
    Xtrain1_shuffle = Xtrain1[idx, :]
    Xtrain2_shuffle = Xtrain2[idx, :]
    ytrain_shuffle = ytrain[idx]
    Xtrain1, Xtrain2, ytrain = torch.from_numpy(Xtrain1_shuffle), torch.from_numpy(Xtrain2_shuffle), torch.from_numpy(
        ytrain_shuffle)
    Xtest1, Xtest2, ytest = torch.from_numpy(Xtest1), torch.from_numpy(Xtest2), torch.from_numpy(ytest)
    return Xtrain1, Xtrain2, ytrain, Xtest1, Xtest2, ytest

# ---------------------------------------------- Model -----------------------------------------------------------------
print('==> Building model..')
net = MLP(10)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01)

# ---------------------------------------------- Train and test---------------------------------------------------------
def train(epochs, iterr):
    for epoch in range(epochs):
        train_loss = 0
        IX1Z1_loss = 0
        IX2Z2_loss = 0
        correct = 0
        total = 0
        net.train()
        inputs1, inputs2, targets, _, _, _ = load_data(True, iterr)  # train data
        num_examples, d1 = inputs1.shape
        inputs1, inputs2, targets = inputs1.float().to(device), inputs2.float().to(device), targets.long().to(device)
        batch_size = args.batchsize
        batch_num = int(num_examples / batch_size)
        for step in range(batch_num):
            x1 = inputs1[step * batch_size: (step + 1) * batch_size, :]
            x2 = inputs2[step * batch_size: (step + 1) * batch_size, :]
            y = targets[step * batch_size: (step + 1) * batch_size]
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            Z1, Z2, outputs = net(x1, x2)
            loss = criterion(outputs, y)
            with torch.no_grad():
                x1_numpy = x1.cpu().detach().numpy()
                k_x1 = squareform(pdist(x1_numpy, 'euclidean'))
                sigma_x1 = np.mean(np.mean(np.sort(k_x1[:, :args.ks], 1)))

                x2_numpy = x2.cpu().detach().numpy()
                k_x2 = squareform(pdist(x2_numpy, 'euclidean'))
                sigma_x2 = np.mean(np.mean(np.sort(k_x2[:, :args.ks], 1)))

                Z1_numpy = Z1.cpu().detach().numpy()
                Z2_numpy = Z2.cpu().detach().numpy()
                k_z1 = squareform(pdist(Z1_numpy, 'euclidean'))
                k_z2 = squareform(pdist(Z2_numpy, 'euclidean'))
                sigma_z1 = np.mean(np.mean(np.sort(k_z1[:, :args.ks], 1)))
                sigma_z2 = np.mean(np.mean(np.sort(k_z2[:, :args.ks], 1)))
            IX1Z1 = calculate_MI(x1, Z1, s_x=sigma_x1 ** 2, s_y=sigma_z1 ** 2)
            IX2Z2 = calculate_MI(x2, Z2, s_x=sigma_x2 ** 2, s_y=sigma_z2 ** 2)
            beta1 = 0.0001
            beta2 = 0.0001
            total_loss = loss + beta1 * IX1Z1 + beta2 * IX2Z2
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item()

            IX1Z1_loss += IX1Z1.item()
            IX2Z2_loss += IX2Z2.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            acc = correct / total

        print("Training:   epoch {}: err={:.4f}".format(epoch, 1 - acc))

        net.eval()
        validate_loss = 0
        test_loss = 0
        v_correct = 0
        t_correct = 0
        v_total = 0
        t_total = 0
        with torch.no_grad():
            _, _, _, v_inputs1, v_input2, v_targets = load_data(True, iterr)  # validation data
            v_inputs1, v_input2, v_targets = v_inputs1.float().to(device), v_input2.float().to(device), \
                                             v_targets.long().to(device)
            _, _, v_outputs = net(v_inputs1, v_input2)

            v_loss = criterion(v_outputs, v_targets)
            validate_loss += v_loss.item()
            _, v_predicted = v_outputs.max(1)
            v_total += v_targets.size(0)
            v_correct += v_predicted.eq(v_targets).sum().item()
            v_acc = v_correct / v_total

            _, _, _, t_inputs1, t_input2, t_targets = load_data(False, iterr)  # test data
            t_inputs1, t_input2, t_targets = t_inputs1.float().to(device), t_input2.float().to(device), \
                                             t_targets.long().to(device)
            _, _, t_outputs = net(t_inputs1, t_input2)

            t_loss = criterion(t_outputs, t_targets)
            test_loss += t_loss.item()
            _, t_predicted = t_outputs.max(1)
            t_total += t_targets.size(0)
            t_correct += t_predicted.eq(t_targets).sum().item()
            t_acc = t_correct / t_total
        print("Validation: epoch {}: acc={:.4f}\terr={:.4f}".format(epoch, v_acc, 1 - v_acc))
        print('Testing:    epoch{}: acc={:.4f}\terr={:.4f}'.format(epoch, t_acc, 1 - t_acc))
    return t_acc


# ---------------------------------------------- main ------------------------------------------------------------------
if __name__ == "__main__":
    t_err_list = []
    result_dir = 'multiview'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = open(result_dir + '/result.txt', 'w')

    for iterr in range(1, 6):
        t_err = train(args.epochs, iterr)
        t_err_list.append(1-t_err)
    result_file.write("t_err_list: %s\n" % (t_err_list))
    result_file.close()
    print('t_err_list:\n', t_err_list)
