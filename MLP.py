import re
import numpy as np
import random


def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


def normalize_d(x):
    return (x + 40.0) / 80.0


def denormalize_y(x):
    return -40.0 + (x * 80.0)


def instant_square_error(y, d):
    e = d - y
    return (e ** 2) / 2


class neuron:
    def __init__(self, dimension, w=None):
        self.dimension = dimension
        self.delta = 0.0
        # if not w == None:
        self.w = w
        # else:
        #     self.w = np.random.randn(self.dimension)
        # self.w = []
        # for i in range(0, dimension):
        #     self.w.append(random.uniform(-5.0, 5.0))

        self.v = 0.0
        self.y = 0.0
        return

    def activation(self, x):
        # print(x)
        # print(self.w)
        self.v = np.dot(x, self.w)
        # for i in range(self.dimension):
        #     self.v += x[i] * self.w[i]
        self.y = stable_sigmoid(self.v)
        return


class MLP:
    def __init__(self, dimension=3, num_layers=3, num_neurons=[10, 10], learn_rate=0.2, max_epoch=100000, tolerate_E=0.01, target_hit_rate=0.92):
        self.layers = []
        self.dimension = dimension
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.learn_rate = learn_rate
        self.max_epoch = max_epoch
        self.train_set = []
        self.test_set = []
        self.tolerate_E = tolerate_E
        self.E = []
        self.target_hit_rate = target_hit_rate
        self.hit_rate_train = 0.0
        self.hit_rate_test = 0.0

    def load_data(self, directory='./output/train4D.txt'):
        xs = []
        with open(directory, 'r') as f:
            for line in f:
                if line != '\n':
                    tokens = re.findall(r'-?\d+\.*\d*', line)
                    x = []
                    x.append(-1.0)
                    for i in range(0, self.dimension):
                        x.append(float(tokens[i]))
                    d = normalize_d(float(tokens[3]))
                    x.append(d)
                    xs.append(np.array(x))
        random.shuffle(xs)
        train_size = int(len(xs) * 2 / 3)
        self.train_set = xs[0: train_size]
        self.test_set = xs[train_size + 1:]
        return

    def load_model(self, directory=None):
        self.layers = []
        if not directory == None:
            # print(directory)
            with open(directory, 'r') as f:
                self.num_layers = int(re.search(r'\d', f.readline()).group())
                tokens = re.findall(r'\d*', f.readline())
                self.num_neurons = []
                for token in tokens:
                    if token == '':
                        continue
                    self.num_neurons.append(int(token))
                # hidden layer
                for ind in range(self.num_layers - 1):
                    l = []
                    f.readline()
                    for i in range(self.num_neurons[ind]):
                        tokens = re.findall(r'-?\d+\.*\d*', f.readline())
                        w = []
                        for token in tokens:
                            w.append(float(token))
                        # print(w)
                        w = np.array(w)
                        if ind == 0:
                            n = neuron(dimension=self.dimension + 1, w=w)
                        else:
                            n = neuron(
                                dimension=self.num_neurons[ind - 1], w=w)
                        l.append(n)
                    self.layers.append(l)
                # output layer
                f.readline()
                tokens = re.findall(r'-?\d+\.*\d*', f.readline())
                w = []
                for token in tokens:
                    w.append(float(token))
                # print(w)
                w = np.array(w)
                self.layers.append(
                    neuron(dimension=self.num_neurons[self.num_layers - 2], w=w))
                # print('Load Done')
        else:
            # hidden layers
            for ind in range(self.num_layers - 1):
                ns = []
                for i in range(self.num_neurons[ind]):
                    if ind == 0:
                        n = neuron(dimension=self.dimension + 1)
                    else:
                        n = neuron(dimension=self.num_neurons[ind - 1])
                    ns.append(n)
                self.layers.append(ns)
            # output layer
            self.layers.append(
                neuron(dimension=self.num_neurons[self.num_layers - 2]))
        return

    def train(self):
        for epoch in range(self.max_epoch):
            if epoch % 2 == 1:
                print(
                    '---Epoch {:d}--- (E = {:.6f}, hit_rate_train = {:.2f}, hit_rate_test = {:.2f})'.format(epoch, self.Eavg(), self.hit_rate_train, self.hit_rate_test))
            random.shuffle(self.train_set)
            for x in self.train_set:
                d = x[self.dimension + 1]
                ys = []
                # feed foward
                # hidden layer
                for l in range(self.num_layers - 1):
                    if l == 0:
                        for n in self.layers[l]:
                            n.activation(x[:-1])
                            ys.append(n.y)
                    else:
                        tmp = []
                        for n in self.layers[l]:
                            n.activation(np.array(ys))
                            tmp.append(n.y)
                        ys = tmp
                # output layer
                self.layers[self.num_layers - 1].activation(np.array(ys))
                # count instant square error
                e = instant_square_error(self.layers[self.num_layers - 1].y, d)
                self.E.append(e)

                # adjust weights
                # output layer
                n = self.layers[self.num_layers - 1]
                n.delta = (d - n.y) * n.y * (1 - n.y)
                for ind in range(len(n.w)):
                    delta_w = self.learn_rate * n.delta * \
                        self.layers[self.num_layers - 2][ind].y
                    n.w[ind] += delta_w

                # hidden layer
                for ind in range(self.num_layers - 2, -1, -1):  # each layer
                    for i in range(self.num_neurons[ind]):      # each neuron
                        n = self.layers[ind][i]
                        # count delta
                        delta = 0.0
                        if ind == self.num_layers - 2:
                            j = self.layers[self.num_layers - 1]
                            delta = j.delta * j.w[i]
                        else:
                            for j in self.layers[ind + 1]:
                                delta += j.delta * j.w[i]
                        n.delta = n.y * (1 - n.y) * delta

                        # not input layer
                        if ind != 0:
                            for j in range(len(n.w)):  # each w
                                delta_w = self.learn_rate * \
                                    n.delta * self.layers[ind - 1][j].y
                                n.w[j] += delta_w

                        # input layer
                        else:
                            for j in range(self.dimension):
                                delta_w = self.learn_rate * n.delta * x[j]
                                n.w[j] += delta_w
            # terminal condition
            if self.test() == True:
                print('Train Done @ epoch {:d}'.format(epoch))
                self.dump_log()
                return True
        print('Failed to fulfill requirements')
        return False

    def test(self):
        # hit_rate
        hit = 0
        random.shuffle(self.test_set)
        for x in self.test_set:
            d = x[self.dimension + 1]
            ys = []
            # feed foward
            # hidden layer
            for l in range(self.num_layers - 1):
                if l == 0:
                    for n in self.layers[l]:
                        n.activation(x[:-1])
                        ys.append(n.y)
                else:
                    tmp = []
                    for n in self.layers[l]:
                        n.activation(np.array(ys))
                        tmp.append(n.y)
                    ys = tmp
            # output layer
            self.layers[self.num_layers - 1].activation(np.array(ys))
            # count instant square error
            e = instant_square_error(self.layers[self.num_layers - 1].y, d)
            if e < self.tolerate_E:
                hit += 1

        self.hit_rate_test = float(hit / len(self.test_set))
        if self.hit_rate_test >= self.target_hit_rate:
            print('Hit Rate OK')
            return True
        random.shuffle(self.train_set)
        hit = 0
        for x in self.train_set:
            d = x[self.dimension + 1]
            ys = []
            # hidden layer
            for l in range(self.num_layers - 1):
                if l == 0:
                    for n in self.layers[l]:
                        n.activation(x[:-1])
                        ys.append(n.y)
                else:
                    tmp = []
                    for n in self.layers[l]:
                        n.activation(np.array(ys))
                        tmp.append(n.y)
                    ys = tmp
            # output layer
            self.layers[self.num_layers - 1].activation(np.array(ys))
            # count instant square error
            e = instant_square_error(self.layers[self.num_layers - 1].y, d)
            if e < self.tolerate_E:
                hit += 1

        self.hit_rate_train = float(hit / len(self.train_set))
        if self.hit_rate_train < self.target_hit_rate:
            # print(float(hit / len(self.train_set)))
            return False

        # E
        if self.Eavg() < self.tolerate_E:
            print('E OK, {:.6f}'.format(self.Eavg()))
            return True

        return False

    def Eavg(self):
        return sum(self.E) / len(self.E)

    def dump_log(self, directory='./output/MLPopt_new.txt'):
        with open(directory, 'w') as f:
            f.write('Layers: {:d}\n'.format(self.num_layers))
            f.write('Hidden Layer Neurons: ')
            for num in self.num_neurons:
                f.write('{:d} '.format(num))
            f.write('\n')
            for i in range(self.num_layers - 1):
                f.write('---Hidden Layer {:d}---\n'.format(i))
                for n in self.layers[i]:
                    for w in n.w:
                        f.write('{:.6f} '.format(w))
                    f.write('\n')
            f.write('---Output Layer---\n')
            for w in self.layers[self.num_layers - 1].w:
                f.write('{:.6f} '.format(w))
            f.write('\n')
        return

    def gety(self, x):
        x.insert(0, -1.0)
        ys = []
        # hidden layer
        for l in range(self.num_layers - 1):
            if l == 0:
                for n in self.layers[l]:
                    n.activation(np.array(x))
                    ys.append(n.y)
            else:
                tmp = []
                for n in self.layers[l]:
                    n.activation(np.array(ys))
                    tmp.append(n.y)
                ys = tmp
        # output layer
        self.layers[self.num_layers - 1].activation(ys)
        return denormalize_y(self.layers[self.num_layers - 1].y)


if __name__ == '__main__':
    model = MLP()
    model.load_data(directory='./output/train4D.txt')
    model.load_model('./output/MLPopt.txt')
    model.max_epoch = 100000
    model.train()
