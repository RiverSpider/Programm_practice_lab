import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import random

class Node:
    def __init__(self, value, color, left=None, right=None, parent=None):
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

class RedBlackTree:
    def __init__(self):
        self.nil = Node(0, 'black')
        self.root = self.nil

    def insert(self, value):
        new_node = Node(value, 'red', self.nil, self.nil, None)
        current = self.root
        while current != self.nil:
            potential_parent = current
            if new_node.value < current.value:
                current = current.left
            else:
                current = current.right
        new_node.parent = potential_parent
        if potential_parent == self.nil:
            self.root = new_node
        elif new_node.value < potential_parent.value:
            potential_parent.left = new_node
        else:
            potential_parent.right = new_node
        self._insert_fixup(new_node)

    def _insert_fixup(self, node):
        while node.parent.color == 'red':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._left_rotate(node.parent.parent)
        self.root.color = 'black'

    def _left_rotate(self, node):
        new_node = node.right
        node.right = new_node.left
        if new_node.left != self.nil:
            new_node.left.parent = node
        new_node.parent = node.parent
        if node.parent == self.nil:
            self.root = new_node
        elif node == node.parent.left:
            node.parent.left = new_node
        else:
            node.parent.right = new_node
        new_node.left = node
        node.parent = new_node

    def _right_rotate(self, node):
        new_node = node.left
        node.left = new_node.right
        if new_node.right != self.nil:
            new_node.right.parent = node
        new_node.parent = node.parent
        if node.parent == self.nil:
            self.root = new_node
        elif node == node.parent.right:
            node.parent.right = new_node
        else:
            node.parent.left = new_node
        new_node.right = node
        node.parent = new_node

def visualize_red_black_tree(tree):
    G = nx.DiGraph()
    queue = [(tree.root, 0, 0)]
    while queue:
        node, level, pos = queue.pop()
        G.add_node(node.value, pos=(pos, -level))
        if node.left != tree.nil:
            queue.insert(0, (node.left, level + 1, pos - 2 ** (5 - level)))
            G.add_edge(node.value, node.left.value)
        if node.right != tree.nil:
            queue.insert(0, (node.right, level + 1, pos + 2 ** (5 - level)))
            G.add_edge(node.value, node.right.value)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, arrows=False)
    plt.show()

def main():
    st.title('Красно-черное дерево визуализатор')
    st.write('Введите числа через пробел для заполнения дерева')
    input_numbers = st.text_input('Введите числа')

    if st.button('Построить дерево'):
        numbers_list = list(map(int, input_numbers.split()))
        tree = RedBlackTree()
        for number in numbers_list:
            tree.insert(number)
        
        st.write('Дерево успешно построено!')
        
        visualize_red_black_tree(tree)

if __name__ == "__main__":
    main()