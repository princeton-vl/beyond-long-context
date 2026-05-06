from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional

from .rules import Rule


class _Node:
    __slots__ = ("children", "fail", "rule_ends", "best_rule_idx", "best_len")

    def __init__(self) -> None:
        self.children: Dict[int, int] = {}
        self.fail: int = 0
        self.rule_ends: List[int] = []
        self.best_rule_idx: int = -1
        self.best_len: int = 0


class RuleAutomaton:
    def __init__(self, rules: List[Rule], alphabet_size: int, precompute_transitions: bool = True):
        self.rules = rules
        self.k = int(alphabet_size)
        if self.k <= 0:
            raise ValueError("alphabet_size must be positive")

        self.nodes: List[_Node] = [_Node()]  # root node index 0

        # Build trie
        for i, r in enumerate(rules):
            cur = 0
            for sym in r.prefix:
                nxt = self.nodes[cur].children.get(sym)
                if nxt is None:
                    nxt = len(self.nodes)
                    self.nodes[cur].children[sym] = nxt
                    self.nodes.append(_Node())
                cur = nxt
            self.nodes[cur].rule_ends.append(i)

        # Failure links BFS
        q: deque[int] = deque()
        root = 0

        for _, child in self.nodes[root].children.items():
            self.nodes[child].fail = root
            q.append(child)

        while q:
            v = q.popleft()
            for sym, u in self.nodes[v].children.items():
                q.append(u)
                f = self.nodes[v].fail
                while f != root and sym not in self.nodes[f].children:
                    f = self.nodes[f].fail
                self.nodes[u].fail = self.nodes[f].children.get(sym, root)

        # Best rule per node (inherit from fail chain)
        for v in self._bfs_order():
            node = self.nodes[v]
            if v == root:
                node.best_rule_idx = -1
                node.best_len = 0
            else:
                f = node.fail
                node.best_rule_idx = self.nodes[f].best_rule_idx
                node.best_len = self.nodes[f].best_len

            for ridx in node.rule_ends:
                L = len(self.rules[ridx].prefix)
                if L > node.best_len:
                    node.best_len = L
                    node.best_rule_idx = ridx

        self.best_rule_idx = [n.best_rule_idx for n in self.nodes]

        self.trans: Optional[List[List[int]]] = None
        if precompute_transitions:
            self.trans = self._build_transitions()

    def _bfs_order(self) -> List[int]:
        order: List[int] = []
        q: deque[int] = deque([0])
        while q:
            v = q.popleft()
            order.append(v)
            for u in self.nodes[v].children.values():
                q.append(u)
        return order

    def _build_transitions(self) -> List[List[int]]:
        trans = [[0] * self.k for _ in range(len(self.nodes))]

        for sym in range(self.k):
            trans[0][sym] = self.nodes[0].children.get(sym, 0)

        for v in self._bfs_order()[1:]:
            node = self.nodes[v]
            f = node.fail
            for sym in range(self.k):
                child = node.children.get(sym)
                trans[v][sym] = child if child is not None else trans[f][sym]
        return trans

    def step_state(self, state: int, sym: int) -> int:
        if self.trans is not None:
            return self.trans[state][sym]

        root = 0
        while state != root and sym not in self.nodes[state].children:
            state = self.nodes[state].fail
        return self.nodes[state].children.get(sym, root)
