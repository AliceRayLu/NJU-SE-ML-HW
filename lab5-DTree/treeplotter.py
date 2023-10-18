# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: treeplotter.py
@time: 2017/2/12
"""
from matplotlib import pyplot as plt
from threading import local

# 全局变量定义

# 决策节点风格
DECISION_NODE = {
    'boxstyle': 'roundtooth',
    'pad': 0.8,
    'fc': 'royalblue'
}
# 叶结点风格
LEAF_NODE = {
    'boxstyle': 'round4',
    'pad': 0.8,
    'fc': 'lime'
}
# 箭头风格
ARROW = {
    'arrowstyle': '<|-'
}
# 全局变量存储
VALUES = local()

def get_leafs_num(tree):
    """
    获得树的叶结点个数，以在绘制树是分配空间
    :param tree: 决策树
    :return: 树的叶结点个数
    """
    if isinstance(tree, dict):
        leafs_num = 0
        root = list(tree.values())[0]
        for node in root.values():
            leafs_num += get_leafs_num(node)
        return leafs_num
    else:
        return 1


def get_tree_deep(tree):
    """
    获得树的深度，以在绘制树的时候分配空间
    :param tree: 决策树
    :return: 树的最大深度
    """
    if isinstance(tree, dict):
        deep = 0
        root = list(tree.values())[0]
        for node in root.values():
            temp_deep = get_tree_deep(node)
            if temp_deep > deep:
                deep = temp_deep
        return deep + 1
    else:
        return 0


def plot_node(ax, text, center_point, parent_point, node_type):
    """
    绘制一个节点
    :param ax: 需要绘制的坐标系
    :param text: 节点上的文字内容
    :param center_point: 节点中心位置
    :param parent_point: 父节点中心位置
    :param node_type: 节点类型（决策节点或叶结点）
    """
    ax.annotate(
        text,
        xy=parent_point,
        xycoords='axes fraction',
        xytext=center_point,
        textcoords='axes fraction',
        va='center',
        ha='center',
        bbox=node_type,
        arrowprops=ARROW
    )


def plot_mid_text(ax, text, center_point, parent_point):
    """
    绘制节点的分支信息（箭头中间的文字）
    :param ax: 需要绘制的坐标轴
    :param text: 文字内容
    :param center_point: 节点中心位置
    :param parent_point:  父节点中心位置
    """
    x_mid = (parent_point[0] - center_point[0]) / 2 + center_point[0]
    y_min = (parent_point[1] - center_point[1]) / 2 + center_point[1]
    ax.text(
        x_mid,
        y_min,
        text,
        backgroundcolor='white',
        va='center',
        ha='center',
    )


def plot_tree(ax, tree, branch='', parent_point=(0.5, 1)):
    """
    绘制一颗树
    :param ax: 需要绘制的坐标系
    :param tree: 需要绘制的决策树
    :param branch: 当前树所属父节点分支
    :param parent_point: 父节点中心位置
    """
    # 计算节点位置，根据页节点树对横向距离进行划分，根据深度对纵向距离进行划分
    leafs_num = get_leafs_num(tree)
    center_point_x = VALUES.used_x + (1 + leafs_num) / 2 / VALUES.leafs_num
    center_point_y = VALUES.used_y
    if isinstance(tree, dict):
        # 如果当前节点不是叶结点， 先循环递归绘制其所有的子节点
        key, root = list(tree.items())[0]
        # 减小纵向偏移
        VALUES.used_y -= 1 / VALUES.tree_deep
        for node_branch, node in root.items():
            plot_tree(ax, node, node_branch, (center_point_x, center_point_y))
        # 纠正纵向偏移
        VALUES.used_y += 1 / VALUES.tree_deep
        # 绘制本节点（即后绘制父节点，这样能防止分支箭头覆盖父节点文字）
        plot_node(ax, key, (center_point_x, center_point_y), parent_point, DECISION_NODE)
        plot_mid_text(ax, branch, (center_point_x, center_point_y), parent_point)
    else:
        # 当前节点是叶结点，直接绘制
        plot_node(ax, tree, (center_point_x, center_point_y), parent_point, LEAF_NODE)
        plot_mid_text(ax, branch, (center_point_x, center_point_y), parent_point)
        # 修改横向偏移
        VALUES.used_x += 1 / VALUES.leafs_num


def create_plot(tree):
    """
    完整绘制一棵决策树
    :param tree:需要绘制的树
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    ax = plt.axes(frameon=False)
    # 消除坐标轴
    ax.set_axis_off()
    # 准备全局变量
    VALUES.leafs_num = get_leafs_num(tree)
    VALUES.tree_deep = get_tree_deep(tree)
    VALUES.used_x = -0.5 / VALUES.leafs_num
    VALUES.used_y = 1
    # 递归绘制决策树
    plot_tree(ax, tree)
    plt.show()
