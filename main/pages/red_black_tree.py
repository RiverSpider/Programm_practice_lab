import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
import time
from enum import Enum
import math

st.set_option('deprecation.showPyplotGlobalUse', False)

from enum import Enum
import networkx as nx
import time
import math

class Color(Enum):
    Black = 'black'
    Red = 'red'

class Position:
	def __init__(self, x: int, y: int) -> None:
		self.x = x
		self.y = y

	def __radd__(self, obj):
		if isinstance(obj, Position):
			return Position(self.x + obj.x, self.y + obj.y)
		elif isinstance(obj, tuple):
			return Position(self.x + obj[0], self.y + obj[1])
		raise TypeError(f'unsupported operand type(s) for +: Position and {type(obj)}')
		
	def __repr__(self) -> str:
		return f'<Position{self.value}>'

	@property
	def value(self) -> tuple:
		return (self.x, self.y)

class Node:
    def __init__(self, father=None) -> None:
        self.color = Color.Black
        self.father: Node | None = father
        self.left: Node | None = None
        self._position: Position | None = None
        self.right: Node | None = None
        self._value: int | None = None

    def __bool__(self) -> bool:
        return bool(self.value) or self.value == 0

    def __eq__(self, obj) -> bool:
        if isinstance(obj, Node):
            return self.value == obj.value if self or obj else self is obj
        elif isinstance(obj, int):
            return self.value == obj
        return False

    def __gt__(self, obj) -> bool:
        if not isinstance(obj, (Node, int)):
            raise ValueError('Object {} not in [Node, int] type'.format(obj))
        return self.value > obj.value if isinstance(obj, Node) else self.value > obj

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __lt__(self, obj) -> bool:
        if not isinstance(obj, (Node, int)):
            raise ValueError('Object {} not in [Node, int] type'.format(obj))
        return self.value < obj.value if isinstance(obj, Node) else self.value < obj

    def __repr__(self) -> str:
        if self:
            return f'<{self.color.name}.Node: {self.value}>'
        elif self.father:
            side = 'Left' if self.is_left else 'Right'
            return f'<{side}.List(father={self.father.value})>'
        return '<Empty root>'

    def __str__(self) -> str:
        return str(self.value) if self else 'n'

    def child(self, value: int):
        return self.left if value < self else self.right

    @property
    def brother(self):
        if not self.father:
            return None
        return self.father.right if self.is_left else self.father.left

    @property
    def children_count(self) -> int:
        return bool(self.right) + bool(self.left)

    @property
    def grandpa(self):
        return self.father.father if self.father else None

    @property
    def is_black(self) -> bool:
        return self.color == Color.Black

    @property
    def is_left(self) -> bool:
        return bool(self.father) and self is self.father.left

    @property
    def is_red(self) -> bool:
        return self.color == Color.Red

    @property
    def position(self) -> Position:
        if not self.father:
            return self._position
        left = (-1)**self.is_left
        pos = self.father.position.value
        return pos + Position(left * 2**(pos[1] - 1), -1)
    
    def set_position(self, count: int):
        height = int(2 * math.log2(count + 1))
        self._position = Position(2**height - 1, height)

    @property
    def uncle(self):
        return self.father.brother if self.father else None

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value if isinstance(value, int) else None
        if self:
            self.left = self.left if self.left != None else Node(father=self)
            self.right = self.right if self.right != None else Node(
                father=self)
        else:
            self.color = Color.Black
            self.left = None
            self.right = None

class RedBlackTree:
    def __init__(self):
        self.root: Node = Node()
        self.nodes: dict[int, Node] = {hash(self.root): self.root}

    def __balance(self, node: Node):
        if node.grandpa and node.father.is_red:
            if node.uncle.is_red:
                node.father.color = Color.Black
                node.uncle.color = Color.Black
                node.grandpa.color = Color.Red
                self.__balance(node.grandpa)
            elif node.father < node.grandpa:
                self.__LLturn(node)
            elif node.father > node.grandpa:
                self.__RRturn(node)
        self.root.color = Color.Black
        self.root.set_position(len(self.nodes))

    def __black_list_case(self, node: Node):
        brother = node.brother
        if not brother:
            return
        if brother.is_black:
            if brother.left.is_black and brother.right.is_black:
                brother.color = Color.Red
                if brother.father.is_red:
                    brother.father.color = Color.Black
                else:
                    self.__black_list_case(node.father)
            elif brother.is_left:
                if brother.right.is_red:
                    self.__RRturn(brother.right.right)
                brother.left.color = Color.Black
                self.__LLturn(brother.left)
            else:
                if brother.left.is_red:
                    self.__LLturn(brother.left.left)
                brother.right.color = Color.Black
                self.__RRturn(brother.right)
        else:
            if brother.is_left:
                self.__LLturn(brother.left)
            else:
                self.__RRturn(brother.right)
            self.__black_list_case(node)

    def __LLturn(self, node: Node):
        if node and node > node.father:
            self.__RRturn(node.right)
        father = node.father
        grandpa = node.grandpa
        uncle = node.uncle
        father_right = father.right
        father.value, grandpa.value = grandpa.value, father.value
        grandpa.right = grandpa.left
        grandpa.left = node
        father.right = uncle
        father.left = father_right
        uncle.father = father
        node.father = grandpa

    def __RRturn(self, node: Node):
        if node and node < node.father:
            self.__LLturn(node.left)
        father = node.father
        grandpa = node.grandpa
        uncle = node.uncle
        father_left = father.left
        father.value, grandpa.value = grandpa.value, father.value
        grandpa.left = grandpa.right
        grandpa.right = node
        father.left = uncle
        father.right = father_left
        uncle.father = father
        node.father = grandpa

    def insert(self, value: int):
        node = self.search(value)
        if node: 
            raise ValueError(f'Value {value} already exists in the tree')
        node.value = value
        node.color = Color.Red
        self.nodes[hash(node.right)] = node.right
        self.nodes[hash(node.left)] = node.left
        self.__balance(node)

    def insert_from(self, values: list[int]):
        for value in values:
            self.insert(value)

    def delete(self, obj: int | Node):
        node = obj if isinstance(obj, Node) else self.search(obj)
        if not node:
            raise ValueError(f'Value {obj} not exists in tree')
        elif node.children_count == 0:
            if node.is_black:
                self.__black_list_case(node)
            self.nodes.pop(hash(node.left))
            self.nodes.pop(hash(node.right))
            node.value = None
        elif node.children_count == 1:
            node_child = node.left or node.right
            node.value, node_child.value = node_child.value, node.value
            self.delete(node_child)
        elif node.children_count == 2:
            max_right_child = node.left
            while max_right_child.right:
                max_right_child = max_right_child.right
            node.value = max_right_child.value
            self.delete(max_right_child)
        self.root.set_position(len(self.nodes))

    def delete_from(self, values: list[int]):
        for value in values:
            self.delete(value)

    def search(self, value: int) -> Node:
        node = self.root
        while node and node != value:
            node = node.child(value)
        return node

    def realize(self, font_size: int = 10, node_size: int = 500):
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes.values())
        g.add_edges_from(self.edges)
        options = {
            "edgecolors": "black",
            "font_color": "white",
            "font_size": font_size,
            "node_color": self.colors,
            "node_size": node_size,
            "width": 4,
        }
        return g, self.positions, options

    @property
    def colors(self) -> list[str]:
        return [node.color.value for node in self.nodes.values()]

    @property
    def edges(self) -> list[tuple[Node]]:
        return [(node, child) for node in self.nodes.values() for child in [node.right, node.left] if node]

    @property
    def positions(self) -> dict[Node, tuple[int]]:
        return {node: node.position.value for node in self.nodes.values()}

session = st.session_state

if 'tree' not in session:
    session.tree = RedBlackTree()

if 'inserted_values' not in session:
    session.inserted_values = []

if 'session_iteration' not in session:
    session.session_iteration = 0

sidebar = st.sidebar

sidebar.subheader('Вставка чисел')
sidebar.text_input(label='', key='insert_field', label_visibility='collapsed')
def clear_insert_text():
    session.new_values = session.insert_field
    session["insert_field"] = ""
sidebar.button(label='Вставить', key='insert_button', on_click=clear_insert_text, use_container_width=True)

sidebar.subheader('Удаление чисел')
sidebar.text_input(
    label='',
    key='values2delete',
    label_visibility='collapsed'
)
def clear_delete_text():
    session.deleting_values = session.values2delete
    session["values2delete"] = ""
sidebar.button(label='Удалить', key='delete_button', on_click=clear_delete_text, use_container_width=True)

sidebar.subheader('Поиск числа')
sidebar.text_input(
    label='',
    key='search_value',
    label_visibility='visible'
)
sidebar.button(label='Поиск', key='search_button', on_click=clear_delete_text, use_container_width=True)

figsize = 12
margins = 0
font_size = 10
node_size = 1000

def visualization():
    tree = session.tree
    g, pos, options = tree.realize(font_size, node_size)
    fig = plt.figure(figsize=[figsize]*2)
    plt.axis('off')
    nx.draw_networkx(g, pos, **options)
    plt.margins(margins)
    st.pyplot(fig)

if session.insert_button:
    try:
        new_values = set([
            int(value) for value in 
                session.new_values.split()
        ])
    except ValueError as e:
        new_values = None
        st.error(f'Неправильный ввод: {e}')

    correct_values = []
    wrong_values = []
    for value in new_values:
        try:
            session.tree.insert(value)
            session.inserted_values.append(value)
            correct_values.append(value)
        except ValueError:
            wrong_values.append(value)
    if correct_values:
        st.success(f'Успешно добавлено: {correct_values}')
    if wrong_values:
        st.warning(f'Ошибка выполнения')

if session.delete_button:
    try:
        values2delete = set([
            int(value) for value in 
                session.deleting_values.split()
        ])
    except ValueError as e:
        values2delete = None
        st.error(f'Ошибка выполнения')

    correct_values = []
    wrong_values = []
    for value in values2delete:
        try:
            session.tree.delete(value)
            session.inserted_values.remove(value)
            correct_values.append(value)
        except ValueError:
            wrong_values.append(value)
    if correct_values:
        st.success(f'Удалено: {correct_values}')
    if wrong_values:
        st.warning(f'Ошибка выполнения')

search_value = session.search_value
if session.search_button:
    try:
        search_value = int(search_value)
        found_node = session.tree.search(search_value)
        if found_node:
            st.success(f'Число {search_value} найдено')
        else:
            st.warning(f'Такого числа нет в дереве')
    except ValueError:
        st.error('Ошибка выполнения')

if session.inserted_values:
    visualization()

