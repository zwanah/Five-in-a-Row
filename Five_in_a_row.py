import sys
import time
from random import choice
import numpy as np
import pygame as pg
from operator import itemgetter
import os
import importlib.util
from copy import deepcopy
import pandas as pd


def visual_coordinates():
    """ Return the visual coordinates (in units of pixels) of
        all stone positions on a five-layer hexagonal board.

        The coordinates are defined for a 640-by-560 window.
    
    Returns
    -------
    XY: ndarray of float
        XY[i, j, 0] = the x-coordinate of position (i, j)
        XY[i, j, 1] = the y-coordinate of position (i, j)
        
        If np.nan in XY[i, j], 
        (i, j) is not a valid position on the board.
    """
    L = 5  # number of layers
    XY = np.full((1 + 2 * L, 1 + 2 * L, 2), np.nan)

    R = (3 ** .5) / 2
    dxNewLayer = [None, -.5, .5, 1, .5, -.5, -1]
    dyNewLayer = [None, R, R, 0, -R, -R, 0]
    dxSameLayer = [None, 1, .5, -.5, -1, -.5, .5]
    dySameLayer = [None, 0, -R, -R, 0, R, R]
    diNewLayer = [None, 1, 1, 0, -1, -1, 0]
    djNewLayer = [None, 0, 1, 1, 0, -1, -1]
    diSameLayer = [None, 0, -1, -1, 0, 1, 1]
    djSameLayer = [None, 1, 0, -1, -1, 0, 1]

    i, j = L, L  # stone positions
    x, y = 0, 0  # visual coordinates
    XY[i, j] = x, y
    for n in range(L):  # build a new layer
        for u in range(1, 7):  # define a corner
            i = (n + 1) * diNewLayer[u] + 5
            j = (n + 1) * djNewLayer[u] + 5
            x = (n + 1) * dxNewLayer[u]
            y = (n + 1) * dyNewLayer[u]
            XY[i, j] = [x, y]
            for v in range(1, n + 1):  # go from the corner clockwise
                i += diSameLayer[u]  # to the next corner
                j += djSameLayer[u]
                x += dxSameLayer[u]
                y += dySameLayer[u]
                XY[i, j] = [x, y]
    XY *= 60
    XY[..., 0] += 320  # x
    XY[..., 1] += 280  # y
    return XY


# 水平情况
def get_pos_row(states, p=1):
    m = states.shape[0]
    strategy_seq = [[0, p, p, p, 0], [p, p, 0, p, p], [0, p, p, p, p], [p, p, p, p, 0], [p, 0, p, p, p],
                    [p, p, p, 0, p], [0, p, 0, p, p, 0], [0, p, p, 0, p, 0]]
    strategy_pos = []

    for row in range(m):
        line = []
        pos = []
        if row <= 5:
            pos = [(row, i) for i in range(row + 6)]  # 该行的位置
            line = [states[i[0]][i[1]] for i in pos]  # 该行的状态
        if row > 5:
            pos = [(row, i) for i in range(row - 5, 11)]
            line = [states[i[0]][i[1]] for i in pos]
        # 长度为5的sub seq
        if line.count(p) >= 3:  # 判断该行是否至少有三个p
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 4)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(5)]
                l = possible_seq[start]
                if l.count(p) >= 3:  # 判断该sub seq是否有至少三个p
                    for i in range(6):
                        if l == strategy_seq[i]:
                            strategy_pos.extend([(list(start)[0], list(start)[1] + i) for i in range(5) if l[i] == 0])
                            break
                        if l.count(p) < 4:  # 若p个数小于4，只判断[0,p,p,p,0]
                            break
        # 长度为6的sub seq
        if line.count(p) >= 3 and line.count(0) >= 3:  # 判断该行是否至少有三个p和三个0
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 5)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(6)]
                l = possible_seq[start]
                if l.count(0) >= 3:  # 判断该sub seq是否有至少三个0
                    for i in range(6, 8):
                        if l == strategy_seq[i]:
                            strategy_pos.extend([(list(start)[0], list(start)[1] + i) for i in range(6) if l[i] == 0])
                            break
    return list(set(strategy_pos))


# 右上斜线情况
def get_pos_right_up(states, p=1):
    m = states.shape[0]
    strategy_seq = [[0, p, p, p, 0], [p, p, 0, p, p], [0, p, p, p, p], [p, p, p, p, 0], [p, 0, p, p, p],
                    [p, p, p, 0, p], [0, p, 0, p, p, 0], [0, p, p, 0, p, 0]]
    strategy_pos = []

    for row in range(m):
        line = []
        pos = []
        if row <= 5:
            pos = [(i, row) for i in range(row + 6)]  # 该行的位置
            line = [states[i[0]][i[1]] for i in pos]  # 该行的状态
        if row > 5:
            pos = [(i, row) for i in range(row - 5, 11)]
            line = [states[i[0]][i[1]] for i in pos]
        # 长度为5的sub seq
        if line.count(p) >= 3:  # 判断该行是否至少有三个p
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 4)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(5)]
                l = possible_seq[start]
                if l.count(p) >= 3:  # 判断该sub seq是否有至少三个p
                    for i in range(6):
                        if l == strategy_seq[i]:
                            strategy_pos.extend([(i + list(start)[0], list(start)[1]) for i in range(5) if l[i] == 0])
                            break
                        if l.count(p) < 4:  # 若p个数小于4，只判断[0,p,p,p,0]
                            break
        # 长度为6的sub seq
        if line.count(p) >= 3 and line.count(0) >= 3:  # 判断该行是否至少有三个p和三个0
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 5)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(6)]
                l = possible_seq[start]
                if l.count(0) >= 3:  # 判断该sub seq是否有至少三个0
                    for i in range(6, 8):
                        if l == strategy_seq[i]:
                            strategy_pos.extend([(i + list(start)[0], list(start)[1]) for i in range(6) if l[i] == 0])
                            break
    return list(set(strategy_pos))


# 左上斜情况
def get_pos_left_up(states, p=1):
    m = states.shape[0]
    strategy_seq = [[0, p, p, p, 0], [p, p, 0, p, p], [0, p, p, p, p], [p, p, p, p, 0], [p, 0, p, p, p],
                    [p, p, p, 0, p], [0, p, 0, p, p, 0], [0, p, p, 0, p, 0]]
    strategy_pos = []
    for row in range(6):
        line = []
        pos = []
        if row == 0:
            for col in range(6):
                pos = [(row + i, i + col) for i in range(states.shape[0] - col)]  # 该行的位置
                line = [states[i[0]][i[1]] for i in pos]
                if line.count(p) >= 3:  # 判断该行是否至少有三个p
                    possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 4)])
                    for (i, start) in zip(range(len(possible_seq)), possible_seq):
                        possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(5)]
                        l = possible_seq[start]
                        if l.count(p) >= 3:  # 判断该sub seq是否有至少三个p
                            for i in range(6):
                                if l == strategy_seq[i]:
                                    strategy_pos.extend(
                                        [(list(start)[0] + i, list(start)[1] + i) for i in range(5) if l[i] == 0])
                                    break
                                if l.count(p) < 4:  # 若p个数小于4，只判断[0,p,p,p,0]
                                    break
                # 长度为6的sub seq
                if line.count(p) >= 3 and line.count(0) >= 3:  # 判断该行是否至少有三个p和三个0
                    possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 5)])
                    for (i, start) in zip(range(len(possible_seq)), possible_seq):
                        possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(6)]
                        l = possible_seq[start]
                        if l.count(0) >= 3:  # 判断该sub seq是否有至少三个0
                            for i in range(6, 8):
                                if l == strategy_seq[i]:
                                    strategy_pos.extend(
                                        [(list(start)[0] + i, list(start)[1] + i) for i in range(6) if l[i] == 0])
                                    break
        else:
            pos = [(row + i, 0 + i) for i in range(states.shape[0] - row)]
            line = [states[i[0]][i[1]] for i in pos]
        # 长度为5的sub seq
        if line.count(p) >= 3:  # 判断该行是否至少有三个p
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 4)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(5)]
                l = possible_seq[start]
                if l.count(p) >= 3:  # 判断该sub seq是否有至少三个p
                    for i in range(6):
                        if l == strategy_seq[i]:
                            strategy_pos.extend(
                                [(list(start)[0] + i, list(start)[1] + i) for i in range(5) if l[i] == 0])
                            break
                        if l.count(p) < 4:  # 若p个数小于4，只判断[0,p,p,p,0]
                            break
        # 长度为6的sub seq
        if line.count(p) >= 3 and line.count(0) >= 3:  # 判断该行是否至少有三个p和三个0
            possible_seq = dict().fromkeys([pos[i] for i in range(len(pos) - 5)])
            for (i, start) in zip(range(len(possible_seq)), possible_seq):
                possible_seq[start] = [states[(pos[i + j])[0]][(pos[i + j])[1]] for j in range(6)]
                l = possible_seq[start]
                if l.count(0) >= 3:  # 判断该sub seq是否有至少三个0
                    for i in range(6, 8):
                        if l == strategy_seq[i]:
                            strategy_pos.extend(
                                [(list(start)[0] + i, list(start)[1] + i) for i in range(6) if l[i] == 0])
                            break
    return list(set(strategy_pos))


# 全部三个子可走地方（包括自己和对方）
def get_pos(states, p):
    return list(set(get_pos_row(states, p) + get_pos_right_up(states, p) + get_pos_left_up(states, p) +
                    get_pos_row(states, -1 * p) + get_pos_right_up(states, -1 * p) + get_pos_left_up(states, -1 * p)))


def get_opponent(player):
    """
  返回对手
  :param player: 玩家
  :return: 返回输入玩家的对手
  """
    opponent = player_o if player == player_x else player_x
    return opponent


player_o = 1  #### ai : black color
player_x = -1  #### human : white color


def ADJ(available_actions):
    adjacent = []
    mm = 11
    all_pos = []
    non_pos = []
    # for row in range(mm):
    #     if row <= 5:
    #         all_pos.extend((row, i) for i in range(row + 6))  # 该行的位置
    #     if row > 5:
    #         all_pos.extend((row, i) for i in range(row - 5, 11))
    for i in range(mm):
        for j in range(mm):
            if abs(i - j) <= 5:
                all_pos.append((i, j))
            else:
                non_pos.append((i, j))

    # print(available_actions)
    # print(list(all_pos))
    moved = list(set(list(all_pos)) - set(list(available_actions)))
    # print(moved)
    # print('niu')
    if not moved:
        if mm % 2 == 1:
            # return [(mm // 2 + 1, mm // 2 + 1)]
            return [(mm // 2, mm // 2)]
    for i in moved:
        row = i[0]
        col = i[1]
        if 0 < col and 0 < row:  # 添加棋子的左上角位置
            adjacent.append((row - 1, col - 1))
        if 0 < row and col < mm - 1:  # 添加棋子的右上角位置
            adjacent.append((row - 1, col))
        if row < mm - 1 and col < mm - 1:  # 添加棋子右下角位置
            adjacent.append((row + 1, col + 1))
        if row < mm - 1 and 0 < col:  # 添加棋子的左下角位置
            adjacent.append((row + 1, col))
        if 0 < col:  # 添加棋子左边位置
            adjacent.append((row, col - 1))
        if col < mm - 1:  # 添加棋子右边位置
            adjacent.append((row, col + 1))
    available_pos = list(set(list(adjacent)) - set(moved) - set(non_pos))
    # print(available_pos)
    # print(list(set(list(adjacent)) - set(moved)))
    return available_pos  # 更新临近点


class Board(object):
    def __init__(self, width=11, height=11, n_in_row=5):
        self.availables = None
        self.width = width
        self.height = height
        # self.states = {}  # 记录当前棋盘的状态，键是位置，值是棋子，这里用玩家来表示棋子类型
        self.n_in_row = n_in_row  # 表示几个相同的棋子连成一线算作胜利
        board = np.zeros((11, 11), int)
        for i in range(11):
            for j in range(11):
                if abs(i - j) <= 5:
                    board[i, j] = 0
                else:
                    board[i, j] = 10
        self.states = board

    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        a1, a2 = np.where(self.states == 0)
        self.availables = list(zip(a1, a2))  # 表示棋盘上所有合法的位置，这里简单的认为空的位置即合法
        # for m in self.availables:
        #     self.states[m[0]][m[1]] = 0  # initialize self.availables
        # print(self.states)

    def update(self, player, move):  # player在move处落子，更新棋盘
        # print("current", move)
        self.states[move[0]][move[1]] = player
        self.availables.remove(move)

    def print_state(self):
        return self.states

    def check_board(self):
        if len(self.availables) == 0:
            return 0, True
        a3, a4 = np.where(self.states != 0)
        moved = list(zip(a3, a4))
        if len(moved) < self.n_in_row * 2 - 1:  ##小于9的时候，不可能有人获胜
            return 0, False
        states = self.states
        n = self.n_in_row
        if len(states[states == 0]) == 0:
            return 0, True
        num_valid = [6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6]
        winner = 0

        for i in range(6):
            for j in range(0, num_valid[i] - 4):
                tmp = np.sum(states[i, j:j + 5])
                if tmp == 5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[i, j - 1] != 1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[i, j + 5] != 1)
                    if l_six and r_six:
                        winner = 1
                elif tmp == -5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[i, j - 1] != -1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[i, j + 5] != -1)
                    if l_six and r_six:
                        winner = -1
                tmp = np.sum(states[j:j + 5, i])
                if tmp == 5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[j - 1, i] != 1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[j + 5, i] != 1)
                    if l_six and r_six:
                        winner = 1
                elif tmp == -5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[j - 1, i] != -1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[j + 5, i] != -1)
                    if l_six and r_six:
                        winner = -1

        for i in range(6, 11):
            for j in range(i - 5, 7):
                tmp = np.sum(states[i, j:j + 5])
                if tmp == 5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[i, j - 1] != 1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[i, j + 5] != 1)
                    if l_six and r_six:
                        winner = 1
                elif tmp == -5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[i, j - 1] != -1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[i, j + 5] != -1)
                    if l_six and r_six:
                        winner = -1
                tmp = np.sum(states[j:j + 5, i])
                if tmp == 5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[j - 1, i] != 1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[j + 5, i] != 1)
                    if l_six and r_six:
                        winner = 1
                elif tmp == -5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and states[j - 1, i] != -1)
                    r_six = (j + 5 >= 11) or ((j + 5 < 11) and states[j + 5, i] != -1)
                    if l_six and r_six:
                        winner = -1

        for offset in range(-5, 6):
            tmp_arr = np.diagonal(states, offset=offset)
            for j in range(0, num_valid[offset + 5] - 4):
                tmp = np.sum(tmp_arr[j:j + 5])
                if tmp == 5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and tmp_arr[j - 1] != 1)
                    r_six = (j + 5 >= len(tmp_arr)) or ((j + 5 < len(tmp_arr)) and tmp_arr[j + 5] != 1)
                    if l_six and r_six:
                        winner = 1
                elif tmp == -5:
                    l_six = (j - 1 < 0) or ((j - 1 >= 0) and tmp_arr[j - 1] != -1)
                    r_six = (j + 5 >= len(tmp_arr)) or ((j + 5 < len(tmp_arr)) and tmp_arr[j + 5] != -1)
                    if l_six and r_six:
                        winner = -1
        if winner == 0:
            return winner, False
        else:
            return winner, True


class Node:
    def __init__(self, board: Board, parent=None, player=1):
        self.state = board  # 当前棋盘状态
        start = time.time()
        if get_pos(board.states, player) != []:
            self.untried_actions = get_pos(board.states, player)
        else:
            self.untried_actions = ADJ(board.availables)  # 可走的地方
        self.parent = parent  # 根节点
        self.player = player  # 谁下棋
        self.children = {}
        self.Q = 0  # 节点最终收益价值
        self.N = 0  # 节点被访问的次数
        self.H = 0  # 节点的启发值

    def weight_func(self, c_param=1.9):
        if self.N != 0:
            # tip： 这里使用了-self.Q 因为子节点的收益代表的是对手的收益
            w = -self.Q / self.N + c_param * np.sqrt(2 * np.log(self.parent.N) / self.N)
        else:
            w = 0.0
        return w

    @staticmethod
    def get_random_action(available_actions):  # random choice 需要加 biased choice
        # print(np.array(available_actions).shape)
        return available_actions[np.random.choice(len(available_actions))]

    def select(self, c_param=1.9):
        weight_max = -float('inf')
        # for child_key in self.children.keys():
        #     tmp = self.children[child_key].weight_func(c_param)
        #     if tmp >= weight_max:
        #         weight_max = tmp
        #         action = child_key
        weights = [child_node.weight_func(c_param) for child_node in self.children.values()]
        action = pd.Series(data=weights, index=self.children.keys()).idxmax()
        next_node = self.children[action]
        return action, next_node

    def expand(self):
        # 从没有尝试的节点中选择
        action = self.untried_actions.pop()
        # 获得当前的节点对应的玩家
        current_player = self.player
        # 获得下一步的局面
        next_board = deepcopy(self.state)

        next_board.init_board()
        next_board.update(current_player, action)

        # 获得下一步的玩家
        next_player = get_opponent(current_player)
        # 扩展出一个子节点
        child_node = Node(next_board, self, next_player)
        self.children[action] = child_node
        return child_node

    def update(self, winner, score):
        self.N += 1
        opponent = get_opponent(self.player)
        if winner == self.player:
            self.Q += 15 / score
        elif winner == opponent:
            self.Q -= 15 / score
        #         elif winner == None:
        #             self.Q += 0.5
        if self.parent:
            self.parent.update(winner, score)

    ### define rollout method
    def rollout(self, depth):
        current_state = deepcopy(self.state)
        current_player = deepcopy(self.player)
        score = depth
        while True:
            winner, is_over = current_state.check_board()
            if is_over:
                break
            if score >= 11:
                return None, score
            if get_pos(current_state.states, current_player) != []:
                available_actions = get_pos(current_state.states, current_player)
            else:

                available_actions = ADJ(current_state.availables)  # 可走的地方
                # print('####')
                # print(current_state.availables)
                # print(available_actions)
            #             available_actions = ADJ(current_state.availables)
            #             available_actions = current_state.availables
            action = Node.get_random_action(available_actions)
            current_state.update(current_player, action)

            current_player = get_opponent(current_player)
            score += 1
        return winner, score

    def is_full_expand(self):
        return len(self.untried_actions) == 0

    def not_root_node(self):
        return self.parent


class MCTS:
    def __init__(self):
        self.root = None
        self.current_node = None
        self.cost_time = 0

    def __str__(self):
        return "monte carlo tree search ai"

    def simulation(self, second=5.):
        start_time = time.time()
        while time.time() - start_time < second:
            leaf_node, depth = self.simulation_policy()
            winner, score = leaf_node.rollout(depth)
            leaf_node.update(winner, score)
        self.cost_time = time.time() - start_time

    def simulation_policy(self):
        current_node = self.current_node
        depth = 1
        while True:
            _, is_over = current_node.state.check_board()
            if is_over:
                break
            if current_node.is_full_expand():
                _, current_node = current_node.select()
            else:
                return current_node.expand(), depth
            depth += 1
        leaf_node = current_node
        return leaf_node, depth

    def take_action(self, current_state, ai_is_black):
        if ai_is_black:
            player = 1
        else:
            player = -1
        if not self.root:  # 第一次初始化
            self.root = Node(current_state, None, player)
            self.current_node = self.root
        else:
            for child_node in self.current_node.children.values():  # 跳转到合适的状态,进而保存之前的记录
                if (child_node.state.states == current_state.states).all():
                    self.current_node = child_node
                    # print("I have used this function")
                    break
                else:  # 游戏重新开始的情况下
                    # print("I have used this function ")
                    self.current_node = Node(current_state, None, player)
        self.simulation(4.95)
        action, next_node = self.current_node.select(0.0)  # 选择概率最大的下一步着子
        self.current_node = next_node  # 跳转到对手状态上,对应上上步的状态跳转,next_node 的孩子点就是我的下一个状态
        return action


def ai_move(board, ai_is_black):
    """ Given the current board, the AI moves according to
        a Monte Carlo tree search and some other tactics.
    
    Parameters
    ----------
    board: ndarray of int
        The current status of the board.
        
    ai_is_black: bool
        If ai_is_black, the AI is the black player,
        otherwise it is the white player.
        
    Returns
    -------
    board: ndarray of int
        The updated status of the board.
    """

    player = 1 if ai_is_black else -1
    ai = MCTS()
    # done = False
    # while not done:
    board.init_board()
    move = ai.take_action(board, ai_is_black)
    # board.init_board()
    if len(move) == 0:
        print('NULL AI SPACE')
        move = board.availables[np.random.choice(len(board.availables))]
    board.update(player, move)

    if ai_is_black:
        print(f"AI black turn: {move}, cost time: {ai.cost_time}(s)")
    else:
        # print("AI white turn", move)
        print(f"AI white turn: {move}, cost time: {ai.cost_time}(s)")
    # print(f"Time : {ai.cost_time}(s)")
    # winner, done = board.check_board()
    return board


def check(board):
    """ Check if a player has won the game.
        There are four possible returns (winner, gameOver):
            (+1,  True): the black player wins
            (-1,  True): the white player wins
            ( 0,  True): draw
            ( 0, False): the game has not yet finished

    Parameters
    ----------
    board: ndarray of int
        The current status of the board.
    
    Returns
    -------
    winner: int
        If the black wins, winner = +1.
        If the white wins, winner = -1.
        If no one wins (yet), winner = 0.
    
    gameOver: bool
        True if a player has won or the game is a draw, otherwise False.
    """

    if len(board[board == 0]) == 0:
        return 0, True
    num_valid = [6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6]
    winner = 0
    for i in range(6):
        for j in range(0, num_valid[i] - 4):
            tmp = np.sum(board[i, j:j + 5])
            if tmp == 5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[i, j - 1] != 1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[i, j + 5] != 1)
                if l_six and r_six:
                    winner = 1
            elif tmp == -5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[i, j - 1] != -1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[i, j + 5] != -1)
                if l_six and r_six:
                    winner = -1
            tmp = np.sum(board[j:j + 5, i])
            if tmp == 5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[j - 1, i] != 1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[j + 5, i] != 1)
                if l_six and r_six:
                    winner = 1
            elif tmp == -5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[j - 1, i] != -1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[j + 5, i] != -1)
                if l_six and r_six:
                    winner = -1

    for i in range(6, 11):
        for j in range(i - 5, 7):
            tmp = np.sum(board[i, j:j + 5])
            if tmp == 5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[i, j - 1] != 1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[i, j + 5] != 1)
                if l_six and r_six:
                    winner = 1
            elif tmp == -5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[i, j - 1] != -1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[i, j + 5] != -1)
                if l_six and r_six:
                    winner = -1
            tmp = np.sum(board[j:j + 5, i])
            if tmp == 5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[j - 1, i] != 1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[j + 5, i] != 1)
                if l_six and r_six:
                    winner = 1
            elif tmp == -5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and board[j - 1, i] != -1)
                r_six = (j + 5 >= 11) or ((j + 5 < 11) and board[j + 5, i] != -1)
                if l_six and r_six:
                    winner = -1

    for offset in range(-5, 6):
        tmp_arr = np.diagonal(board, offset=offset)
        for j in range(0, num_valid[offset + 5] - 4):
            tmp = np.sum(tmp_arr[j:j + 5])
            if tmp == 5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and tmp_arr[j - 1] != 1)
                r_six = (j + 5 >= len(tmp_arr)) or ((j + 5 < len(tmp_arr)) and tmp_arr[j + 5] != 1)
                if l_six and r_six:
                    winner = 1
            elif tmp == -5:
                l_six = (j - 1 < 0) or ((j - 1 >= 0) and tmp_arr[j - 1] != -1)
                r_six = (j + 5 >= len(tmp_arr)) or ((j + 5 < len(tmp_arr)) and tmp_arr[j + 5] != -1)
                if l_six and r_six:
                    winner = -1
    if winner == 0:
        return winner, False
    else:
        return winner, True


def main(you_are_black, opponent_name=None):
    """ Let you play five-in-arow against your AI or
        let your AI play against another group's AI.
    
    Parameters
    ----------
    you_are_black: bool
        If you_are_black, you are the black player,
        otherwise you are the white player.
    
    opponent_name: str or None
        If it is None, you play against your AI.
        If it is str, this function imports opponent_name so that
        your ai_move plays against opponent_name.ai_move.
    """

    def get_wins_title():
        who_play = {}
        if opponent_name is None:
            who_play['black'] = 'You win!' if you_are_black else 'AI wins!'
            who_play['white'] = 'AI wins!' if you_are_black else 'You win!'
        else:
            who_play['white'] = 'AI wins!'
            who_play['black'] = 'AI wins!'
        return who_play

    def get_func(path, func_name):
        if ".py" not in path:
            path += '.py'
        if not os.path.isfile(path):
            raise IOError
        # 根据提供的目标文件路径，返回模块的说明  spec_from_file_location(module_name, file_path)
        module_desc = importlib.util.spec_from_file_location("_", path)
        # 根据传入的模块说明返回引入的模块 module_from_spec（模块说明）
        moudle_spec = importlib.util.module_from_spec(module_desc)
        # exec_module执行加载方法
        module_desc.loader.exec_module(moudle_spec)
        if hasattr(moudle_spec, func_name):
            return getattr(moudle_spec, func_name)

    # # board 初始化
    # board = np.zeros((11, 11), int)
    # for i in range(11):
    #     for j in range(11):
    #         if abs(i - j) <= 5:
    #             board[i, j] = 0
    #         else:
    #             board[i, j] = 10
    board = Board()
    board.init_board()

    xy = visual_coordinates()
    wins_title = get_wins_title()

    # 点击得到棋子
    def find_pos(pos_x, pos_y):
        for i in range(11):
            for j in range(11):
                if ((pos_x - xy[i, j, 0]) ** 2 + (pos_y - xy[i, j, 1]) ** 2) <= 400:
                    return i, j
        return -1, -1

    # 判定是否点击合法
    def check_over_pos(xi, yj):
        if xi < 0 or yj < 0:
            return False
        elif board.states[xi][yj] == 0:
            return True

    # 更新所有11*11的矩阵可视化
    def update_graph(g=board.states):
        for i in range(11):
            for j in range(11):
                if g[i, j] == 1:
                    pg.draw.circle(surface, black_piece_color, xy[i][j], 9)
                    pg.draw.circle(surface, line_color, xy[i][j], 9, width=1)
                elif g[i, j] == -1:
                    pg.draw.circle(surface, white_piece_color, xy[i][j], 9)
                    pg.draw.circle(surface, line_color, xy[i][j], 9, width=1)

    # 游戏结束打印
    def text_result(winner, draw):
        if draw and winner == 0:
            text = fnt.render("Draw!", True, black_piece_color)
            surface.blit(text, (0, 0))
            # pg.display.update()
            return True
        else:
            if winner == 1:
                text = fnt.render(wins_title['black'], True, black_piece_color)
                surface.blit(text, (0, 0))
                # pg.display.update()
                return True
            elif winner == -1:
                text = fnt.render(wins_title['white'], True, white_piece_color)
                surface.blit(text, (0, 0))
                # pg.display.update()
                return True
        return False

    # 关闭回收 + 锁死
    def over_procedure():
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

    # check()封装 （封装了可视化功能 + 结束锁死pygame机制）
    def over_check():
        w, d = check(board.states)
        is_over = text_result(w, d)
        pg.display.update()
        if is_over:
            over_procedure()

    # 人类下棋
    def human_turn():
        valid = False
        while valid is not True:
            event = pg.event.wait()
            if event.type == pg.MOUSEBUTTONDOWN:
                x, y = find_pos(event.pos[0], event.pos[1])
                if check_over_pos(x, y):
                    board.states[x, y] = 1 if you_are_black else -1
                    update_graph()
                    valid = True
            elif event.type == pg.QUIT:
                pg.quit()
                sys.exit()

    # AI 下棋（
    def my_ai_turn(g, ai_is_black=(not you_are_black)):
        g1 = ai_move(g, ai_is_black)
        update_graph(g1.states)
        return g1

    def opponent_ai_turn(g, ai_is_black):
        func = get_func(opponent_name, "ai_move")
        g1 = func(g, ai_is_black)
        update_graph(g1.states)
        return g1

    pg.init()
    surface = pg.display.set_mode((640, 600))
    fnt = pg.font.Font(None, 38)
    board_color = [0xe7, 0xb9, 0x41]
    line_color = [0, 0, 0]
    black_piece_color = [0, 0, 0]
    white_piece_color = [255, 255, 255]
    surface.fill(board_color)
    for i in range(5):
        pg.draw.line(surface, line_color, xy[i, 0], xy[i, 5 + i])
        pg.draw.line(surface, line_color, xy[0, i], xy[5 + i, i])
        pg.draw.line(surface, line_color, xy[5 - i, 0], xy[10, 5 + i])
    for i in range(6):
        pg.draw.line(surface, line_color, xy[5 + i, i], xy[5 + i, 10])
        pg.draw.line(surface, line_color, xy[i, 5 + i], xy[10, 5 + i])
        pg.draw.line(surface, line_color, xy[0, i], xy[10 - i, 10])

    pg.display.set_caption('Our Game')
    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        if opponent_name is None:
            if you_are_black:
                human_turn()
                over_check()
                board = my_ai_turn(board)
                over_check()
            else:
                board = my_ai_turn(board)
                # print(board)
                over_check()
                human_turn()
                over_check()
        else:
            if you_are_black:
                board = my_ai_turn(board, True)
                over_check()
                board = opponent_ai_turn(board, False)
                over_check()
            else:
                board = opponent_ai_turn(board, True)
                over_check()
                board = my_ai_turn(board, False)
                over_check()


if __name__ == '__main__':
    main(you_are_black=False    ·, opponent_name=None)



