import random
import tensorflow as tf
import numpy as np
import pickle
import os
from src.utils.movesprep import data_path

from src.utils import moves
from src.utils import movesprep
from src.utils import representations


class Search:

    def __init__(self, left_moves, right_moves):
        self.lm = left_moves
        self.rm = right_moves

    def highest_among(self, values):
        best_state = values[0][0]
        best_score = values[0][1]

        for nxt in values:
            if nxt[1] > best_score:
                best_state = nxt[0]
                best_score = nxt[1]

        return best_state

    def rollout(self, state):
        """
        :param state: game state from which playout happens
        :return: number of steps during the playout
        """
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        if int(len(reachable)) == 0:
            return 0

        # random strategy
        reachable = list(reachable)
        random.shuffle(reachable)
        return 1 + self.rollout(moves.random_tile(reachable[0]))

    def best_move(self, state, search_iters):
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        nxts = []
        for r in reachable:
            score = 0
            for i in range(search_iters):
                score += self.rollout(moves.random_tile(r))
            nxts.append((r, score))
        return self.highest_among(nxts)

    def play_game(self, per_move, verbose=False):
        state = [0, 0, 0, 0]
        state = moves.random_tile(state)
        reachable = moves.possible_moves(state, self.lm, self.rm).values()

        lvl = 0
        while len(reachable) > 0:
            if verbose:
                print('move', lvl, state)
            lvl += 1
            n = self.best_move(state, per_move)
            state = moves.random_tile(n)
            reachable = moves.possible_moves(state, self.lm, self.rm).values()


class MCTS(Search):

    def __init__(self, left_moves, right_moves, exploration_coefficient):
        super().__init__(left_moves, right_moves)
        self.scoresum = {}
        self.count = {}
        self.c = exploration_coefficient
        pass

    def save_tree(self):
        with open(os.path.join(data_path(), 'mcts-counts.pickle'), 'wb') as file:
            pickle.dump(self.count, file)
        with open(os.path.join(data_path(), 'mcts-sums.pickle'), 'wb') as file:
            pickle.dump(self.scoresum, file)

    def restore_tree(self):
        with open(os.path.join(data_path(), 'mcts-counts.pickle'), 'rb') as file:
            self.count = pickle.load(file)
        with open(os.path.join(data_path(), 'mcts-sums.pickle'), 'rb') as file:
            self.scoresum = pickle.load(file)

    def avgscore(self, state):
        if tuple(state) not in self.count:
            return 0
        return self.scoresum[tuple(state)] / 10. / self.count[tuple(state)]

    def ucb(self, state, parent):
        if tuple(state) not in self.count:
            return 1000.
        n_parent = self.count[tuple(parent)]
        n_state = self.count[tuple(state)]
        exploration_term = self.c * np.sqrt(np.log(n_parent) / n_state)
        return self.avgscore(state) + exploration_term

    def highest_ucb(self, reachable, parent, verb=False):
        ucbs = []
        for nxt in reachable:
            score = self.ucb(nxt, parent)
            ucbs.append((nxt, score))

        if verb:
            print(parent, ucbs, self.highest_among(ucbs))
        return self.highest_among(ucbs)

    def search(self, state):
        if tuple(state) not in self.count:
            self.count[tuple(state)] = 1
            self.scoresum[tuple(state)] = 0
            res = self.rollout(state)
            self.scoresum[tuple(state)] = res
            return res + 1

        self.count[tuple(state)] += 1
        reachable = moves.possible_moves(state, self.lm, self.rm).values()

        if len(reachable) == 0:
            return 0

        chosen = self.highest_ucb(reachable, state)

        if tuple(chosen) not in self.count:
            self.count[tuple(chosen)] = 0
            self.scoresum[tuple(chosen)] = 0
        self.count[tuple(chosen)] += 1

        nxt = moves.random_tile(chosen)
        res = self.search(nxt)

        self.scoresum[tuple(chosen)] += res - 1
        self.scoresum[tuple(state)] += res
        return 1 + res

    def best_move(self, state, search_iters):
        for i in range(search_iters):
            self.search(state)

        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        state_score = []
        for r in reachable:
            state_score.append((r, self.avgscore(r)))
        return self.highest_among(state_score)

    def play_game(self, search_iters, verbose=False):
        state = [0, 0, 0, 0]
        state = moves.random_tile(state)
        reachable = moves.possible_moves(state, self.lm, self.rm).values()

        lvl = 0
        while len(reachable) > 0:
            if verbose:
                print('move', lvl, state)
            lvl += 1

            chosen = self.best_move(state, search_iters)
            state = moves.random_tile(chosen)
            reachable = moves.possible_moves(state, self.lm, self.rm).values()

    def dataset(self, nn):
        data = []
        batch_size = 10000
        for state, cnt in self.count.items():
            data.append([state, self.scoresum[state] / cnt / 100.])
        x_train = []
        y_train = []
        i = 0
        while i < len(data):
            batch = data[i:i + batch_size]
            x_train.append([list(s[0]) for s in batch])
            y_train.append([[s[1]] for s in batch])
            i += batch_size
        return x_train, y_train

    def policy_dataset(self):
        data = []
        batch_size = 1000
        dct = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        for state, cnt in self.count.items():
            reachable = moves.possible_moves(state, self.lm, self.rm)  # ri, le, dn, up
            scores = [0.,0.,0.,0.]
            no_data = False
            i = 0
            j = 0
            mx = -1
            for dire, st in reachable.items():
                if tuple(st) not in self.count:
                    no_data = True
                    break
                else:
                    if mx < self.scoresum[tuple(st)] / self.count[tuple(st)]:
                        mx = self.scoresum[tuple(st)] / self.count[tuple(st)]
                        i = j
                j += 1
            if mx != -1 and not no_data:
                scores[i] = 1.
                data.append([list(state), list(scores)])

            if no_data:
                continue

        print(len(data))
        print(data[0])

        x_train = []
        y_train = []

        i = 0
        while i < len(data):
            batch = data[i:i + batch_size]
            x_train.append([representations.chanelled_flat(representations.tiles_2d(s[0])) for s in batch])
            y_train.append([s[1] for s in batch])
            i += batch_size

        return x_train, y_train


class NetworkMCTS(MCTS):

    def __init__(self, left_moves, right_moves, exploration_coefficient, neural_network, discount):
        super().__init__(left_moves, right_moves, exploration_coefficient)
        self.nn = neural_network
        self.batch = []
        self.gamma = discount
        self.data = []
        self.lrs = []

    def rollout(self, state):
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        if int(len(reachable)) == 0:
            return 0

        tiles = representations.tiles_2d(state)
        #channelled = representations.chanelled_tiles(tiles)
        pred = self.nn.predict(np.array([tiles]))[0]
        return pred

    def sim_pol(self, net, state):
        r = moves.possible_moves(state, self.lm, self.rm)
        if len(r) == 0:
            return 0
        x = np.array([representations.chanelled_tiles(representations.tiles_2d(state))])
        x = np.array([representations.tiles_2d(state)])
        y = net.predict(x)[0]
        #print(y)
        act = 'left'
        mx = -1
        dct = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        for dire, st in r.items():
            if mx < y[dct[dire]]:
                mx = y[dct[dire]]
                act = dire
        print(act[0], end='')
        state = r[act]
        state = moves.random_tile(state)
        return 1 + self.sim_pol(net, state)

    def best_move_net(self, state, net):
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        if int(len(reachable)) == 0:
            return None
        x = []
        for r in reachable:
            #x.append(representations.chanelled_tiles(representations.tiles_2d(r)))
            x.append(representations.tiles_2d(r))
        x = np.array(x)
        y = net.predict(x)
        return self.highest_among(list(zip(reachable, y)))

    def play_game_net_policy(self, net):
        state = [0,0,0,0]
        state = moves.random_tile(state)
        return self.sim_pol(net, state)

    def simulate_game_net(self, state, nn):
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        if int(len(reachable)) == 0:
            return 0
        x = []
        for r in reachable:
            #x.append(representations.chanelled_tiles(representations.tiles_2d(r)))
            x.append(representations.tiles_2d(r))
        x = np.array(x)
        y = nn.predict(x)
        chosen = self.highest_among(list(zip(reachable, y)))
        nxt = moves.random_tile(chosen)
        return self.simulate_game_net(nxt, nn) + 1

    def play_network_only(self, nn):
        state = [0, 0, 0, 0]
        state = moves.random_tile(state)
        return self.simulate_game_net(state, nn)

    def alternative_play(self, search_iters, target=5, verbose=False, make_batch=False):
        self.batch.clear()
        state = [0, 0, 0, 0]
        state = moves.random_tile(state)
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        batch = []

        lvl = 0
        final_score = -1.

        while len(reachable) > 0:
            if representations.contains(state, target):
                final_score = 1.
                break

            if make_batch:
                batch.append([state, lvl, 0.])

            if verbose:
                print('move', lvl, state)
            lvl += 1

            chosen = self.best_move(state, search_iters)
            if make_batch:
                batch.append([chosen, lvl, 0.])
            state = moves.random_tile(chosen)
            reachable = moves.possible_moves(state, self.lm, self.rm).values()

        print('moves', lvl)
        if make_batch:
            batch.append([state, lvl, final_score])
            for i in range(len(batch) - 2, -1, -1):
                batch[i][2] = final_score
            self.batch += batch

    def play_game(self, search_iters, verbose=False, make_batch=False):
        self.batch.clear()
        state = [0, 0, 0, 0]
        state = moves.random_tile(state)
        reachable = moves.possible_moves(state, self.lm, self.rm).values()
        batch = []

        lvl = 0
        while len(reachable) > 0:
            if make_batch:
                batch.append([state, lvl, 0.])

            if verbose:
                print('move', lvl, state)
            lvl += 1

            chosen = self.best_move(state, search_iters)
            state = moves.random_tile(chosen)
            reachable = moves.possible_moves(state, self.lm, self.rm).values()

        print('moves', lvl)
        if make_batch:
            batch.append([state, lvl, 0.])
            for i in range(len(batch) - 2, -1, -1):
                batch[i][2] = 1 + self.gamma * batch[i + 1][2]
            self.batch += batch

    def fit_batch(self, epochs, lr):
        x_train = []
        y_train = []
        for example in self.batch:
            tiles = representations.tiles_2d(example[0])
            #channelled = representations.chanelled_tiles(tiles)
            x_train.append(tiles)
            y_train.append([example[2]])
        x_train = np.array([x_train])
        y_train = np.array([y_train])
        self.nn.fit(x_train, y_train, epochs, lr, False)

    def battle(self, nn):
        nnscore = 0
        selfscore = 0
        for i in range(10):
            nnscore += self.play_network_only(nn)
            selfscore += self.play_network_only(self.nn)
        print('Scores', round(selfscore / nnscore, 2), selfscore/10, nnscore/10)
        if selfscore / nnscore < 1.05:
            return nn, False
        else:
            print('Updated nets', round(selfscore / nnscore, 2), selfscore/10, nnscore/10)
            return self.nn, True

    def normalize_batch(self):
        for i in range(len(self.batch)):
            self.batch[i][-1] /= 1000.

    def avgnetscores(self, nns):
        sc = []
        for nn in nns:
            sm = 0.
            for i in range(10):
                sm += self.play_network_only(nn)
            sc.append(sm/10)
        return sc

    def train(self, rounds, plays_per_round, batch_lr_discount, initial_lr, batch_ignore_threshold, target):
        self.data = []
        self.lrs = []

        for r in range(rounds):
            print('Round', r + 1, 'begins')
            prevdata = self.data.copy()
            prevlrs = self.lrs.copy()
            for i in range(int(len(self.data))):
                self.lrs[i] *= batch_lr_discount

            while len(self.lrs) > 0 and self.lrs[0] < batch_ignore_threshold:
                self.data = self.data[1:]
                self.lrs = self.lrs[1:]

            for p in range(plays_per_round):
                #self.play_game(100, verbose=False, make_batch=True)
                self.alternative_play(100, target, verbose=False, make_batch=True)

            self.normalize_batch()
            self.data.append(self.batch)
            self.lrs.append(initial_lr)

            nns = []
            setups = [(50, 1., 10), (10, 1., 5), (35, 1., 1), (12, 1., 2), (5, 3., 3)]  # (epochs, lr, datanum)
            for i in range(5):
                nns.append(self.nn.cpy())

            nnself = self.nn.cpy()
            for i, nn in enumerate(nns):
                self.nn = nns[i]
                for j in range(len(self.data) - 1, max(-1, len(self.data) - 1 - setups[i][2]), -1):
                    self.batch = self.data[j]
                    self.fit_batch(setups[i][0], self.lrs[j] * setups[i][1])

            nns.append(nnself)
            avgs = self.avgnetscores(nns)
            print('avg scores', avgs)
            best = 0.
            better = True
            for i, a in enumerate(avgs):
                if a > best:
                    if i == len(avgs) - 1:
                        better = False
                    best = a
                    self.nn = nns[i]

            if not better:
                self.data = prevdata
                self.lrs = prevlrs
            else:
                print('upgraded', best, avgs[-1])

            with open(os.path.join(data_path(), 'mcts-netH1.pickle'), 'wb') as file:
                pickle.dump(self.nn, file)


def play_tf(net, lm, rm):
    state = [0,0,0,0]
    state = moves.random_tile(state)
    return sim_tf(state, net, lm, rm)


def sim_tf(state, net: tf.keras.models.Sequential, lm, rm):
    r = moves.possible_moves(state, lm, rm)
    if len(r) == 0:
        return 0
    x = representations.chanelled_tiles(representations.tiles_2d(state))
    y = net.predict(np.array([x]), batch_size=1)[0]
    act = 'left'
    mx = -1
    dct = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    for dire, st in r.items():
        if mx < y[dct[dire]]:
            mx = y[dct[dire]]
            act = dire
    print(act[0], end='')
    state = r[act]
    state = moves.random_tile(state)
    return 1 + sim_tf(state, net, lm, rm)


if __name__ == '__main__':

    lm, rm = movesprep.load_moves()
    import src.mllib.models as models
    import src.mllib.layers as layers

    '''
    net = models.Sequential()
    #net.add_layer(layers.Convolutional(4, 4, 16, 16, 2, 'tanh'))
    #net.add_layer(layers.Convolutional(3, 3, 16, 16, 2, 'tanh'))
    net.add_layer(layers.Flatten(4, 4, 1))
    net.add_layer(layers.Dense(16, 32, 'tanh'))
    net.add_layer(layers.Dense(32, 64, 'tanh'))
    net.add_layer(layers.Dense(64, 128, 'tanh'))
    net.add_layer(layers.Dense(128, 64, 'tanh'))
    net.add_layer(layers.Dense(64, 32, 'tanh'))
    net.add_layer(layers.Dense(32, 1, 'tanh'))
    '''

    with open(os.path.join(data_path(), 'mcts-netH0.pickle'), 'rb') as file:
        net = pickle.load(file)

    nmcts = NetworkMCTS(lm, rm, 1., net, 0.997)
    nmcts.train(100, 5, 0.9, 0.001, 0.001, 8)



'''
    tree = MCTS(lm, rm, 10.)
    #tree.restore_tree()
    print(len(tree.count))
    nmcts = NetworkMCTS(lm, rm, 0., net, 0.1)   # only for some functions

    x_train, y_train = tree.dataset(net)

    epochs = 1
    lr = 0.001

    for e in range(epochs):
        print('Epoch', e + 1)
        b = 0
        for X, Y in list(zip(x_train, y_train)):
            b += 1
            X = np.array([representations.chanelled_tiles(representations.tiles_2d(x)) for x in X])
            Y = np.array(Y)
            net.predict(X)
            net.backward(Y, lr)
            nmcts.count = tree.count
            nmcts.scoresum = tree.scoresum
            print('Batch', b, 'loss:', net.summary_loss(np.array([X]), np.array([Y])), 'moves test:', nmcts.play_network_only(net))
    print('Finished')
    with open(os.path.join(data_path(), 'net-free-train.pickle'), 'wb') as file:
        pickle.dump(net, file)
    
    net.add_layer(layers.Convolutional(4, 4, 1, 8, 2, 'sigmoid'))
    net.add_layer(layers.Convolutional(3, 3, 8, 16, 2, 'sigmoid'))
    net.add_layer(layers.Flatten(2, 2, 16))
    net.add_layer(layers.Dense(64, 128, 'sigmoid'))
    net.add_layer(layers.Dense(128, 64, 'sigmoid'))
    net.add_layer(layers.Dense(64, 4, 'sigmoid'))
    

    tree = MCTS(lm, rm, 0.)
    tree.restore_tree()
    print(len(tree.count))
    nmcts = NetworkMCTS(lm, rm, 0., net, 0.1)   # only for some functions

    #x_train, y_train = tree.policy_dataset()
    #tp = (x_train, y_train)
    #with open(os.path.join(data_path(), 'policy-dataset-flat.pickle'), 'wb') as file:
    #    pickle.dump(tp, file)

    with open(os.path.join(data_path(), 'policy-dataset.pickle'), 'rb') as file:
        x_train, y_train = pickle.load(file)

    x_train = x_train[:-1]
    y_train = y_train[:-1]
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    
    nmcts.nn = net
    print(' play', nmcts.play_game_net_policy(net))
    print('loss before', net.summary_loss(x_train, y_train))
    net.fit(x_train, y_train, 1, 0.01, verbose=True)
    print(' play after', nmcts.play_game_net_policy(net))
    net.fit(x_train, y_train, 1, 0.01, verbose=True)
    print(' play after', nmcts.play_game_net_policy(net))
    net.fit(x_train, y_train, 1, 0.01, verbose=True)
    print(' play after', nmcts.play_game_net_policy(net))
    net.fit(x_train, y_train, 1, 0.01, verbose=True)
    print(' play after', nmcts.play_game_net_policy(net))
    net.fit(x_train, y_train, 1, 0.01, verbose=True)
    print(' play after', nmcts.play_game_net_policy(net))
    
    
    with open(os.path.join(data_path(), 'net-policy-train.pickle'), 'wb') as file:
        pickle.dump(net, file)
    


    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Conv2D(32, (2, 2), activation='sigmoid'))
    net.add(tf.keras.layers.Conv2D(32, (2, 2), activation='sigmoid'))
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(128, 'sigmoid'))
    net.add(tf.keras.layers.Dense(64, 'sigmoid'))
    net.add(tf.keras.layers.Dense(4, 'sigmoid'))

    opt = tf.keras.optimizers.Adam()
    net.compile(opt, loss=tf.keras.losses.mse)

    x_tr = []
    y_tr = []
    for b in x_train:
        x_tr += list(b)
    for b in y_train:
        y_tr += list(b)

    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)

    print(play_tf(net, lm, rm))
    net.fit(x_tr, y_tr, epochs=30)

    print(' moves', play_tf(net, lm, rm))
    print(' moves', play_tf(net, lm, rm))
    print(' moves', play_tf(net, lm, rm))
    print(' moves', play_tf(net, lm, rm))
    print(' moves', play_tf(net, lm, rm))

    
    TRAINING MCTS
    
    tree = MCTS(lm, rm, 5.)
    tree.restore_tree()
    for i in range(10):
        print('MCTS playing')
        tree.play_game(100, True)
        tree.save_tree()
        print('seen states:', len(tree.count))
        print('Finished')
'''