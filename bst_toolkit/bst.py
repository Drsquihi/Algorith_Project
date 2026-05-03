from __future__ import annotations
from typing import Optional, List
from .node import TrialNode
from collections import deque


class BST:
    """
    Binary Search Tree keyed by trial score.
    Left child < parent < right child (BST property).
    All operations are O(h) where h = tree height.
    """

    def __init__(self) -> None:
        self.root = None
        self._size = 0

    # ── Public methods ─────────────────────────────────────────────

    def insert(self, score: float, params: dict) -> None:
        """
        Insert a new trial into the BST.
        If a node with the same score already exists, KEEP THE EXISTING
        PARAMS (first-inserted wins). The duplicate is silently ignored.

        Complexity: O(h) average.

        Note: to minimise collisions in practice, round scores to 6 decimals
              before calling insert, e.g. registry.add_trial(round(score, 6), params).
        """
        score = round(score, 6)

        if self.search(score) is None:
            self.root = self._insert(self.root, score, params)
            self._size += 1

    def delete(self, score: float) -> None:
        """
        Delete the node with the given score.

        Apply the correct case (0, 1, or 2 children).
        Complexity: O(h).
        """
        score = round(score, 6)
        self.root, deleted = self._delete(self.root, score)

        if deleted:
            self._size -= 1

    def search(self, score: float) -> Optional[TrialNode]:
        """
        Return the node with this score, or None if not found.

        Complexity: O(h).
        """
        
        score = round(score, 6)
        return self._search(self.root, score)

    def find_min(self) -> Optional[TrialNode]:
        """
        Return the node with the lowest score (leftmost node).

        Complexity: O(h).
        """
        node = self._find_min(self.root)
        return node

    def find_max(self) -> Optional[TrialNode]:
        """
        Return the node with the highest score (rightmost node).

        Complexity: O(h).
        """
        node = self.root

        if node is None:
            return None
        
        while node.right is not None:
            node = node.right
        
        return node

    def height(self) -> int:
        """
        Return the height of the tree (0 for an empty tree).

        Complexity: O(n).
        """
        return self._height(self.root)

    def is_balanced(self) -> bool:
        """
        Return True if the tree is balanced:
        |height(left) - height(right)| <= 1 at every node.
        """
        return self._check_balanced(self.root) != -1

    def __len__(self) -> int:
        return self._size

    # ── Traversals ─────────────────────────────────────────────

    def inorder(self) -> List[TrialNode]:
        """
        In-order traversal: Left → Node → Right.
        Returns nodes sorted by score ASCENDING.

        Complexity: O(n).
        """
        result = []
        self._inorder(self.root,result)
        return result

    def preorder(self) -> List[TrialNode]:
        """
        Pre-order traversal: Node → Left → Right.
        Returns the root first - used to serialise/copy the tree.

        Complexity: O(n).
        """
        result = []
        self._preorder(self.root, result)
        return result

    def postorder(self) -> List[TrialNode]:
        """
        Post-order traversal: Left → Right → Node.
        Returns the root last - used to delete the tree safely.

        Complexity: O(n).
        """
        
        result = []
        self._postorder(self.root, result)
        return result

    def level_order(self) -> List[List[TrialNode]]:
        """
        Breadth-first traversal: level by level, left to right.

        Uses a queue (collections.deque), NOT recursion.
        Returns a list of levels: [[root], [level1_nodes], ...].

        Complexity: O(n).
        """
        if self.root is None:
            return []

        result = []
        queue = deque([self.root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node)

                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)

            result.append(level)

        return result
    # ── Private helpers ─────────────────────────────────────────────

    def _insert(self, node, score, params):
        if node is None:
            return TrialNode(score,params)
        if score == node.score:
            return node
        if score < node.score:
            node.left = self._insert(node.left, score, params)
        else :
            node.right = self._insert(node.right, score, params)
        return node

    def _delete(self, node, score):
        """
        Recursive delete helper. Returns (node, was_deleted: bool).

        Three cases:
        1. Leaf (no children): return None, True
        2. One child: return the existing child, True
        3. Two children:
           - find in-order successor (min of right subtree)
           - copy its value into this node
           - delete it from the right subtree
        """

        if node is None:
            return None, False

        if score < node.score:
            node.left, deleted = self._delete(node.left, score)
            return node, deleted

        if score > node.score:
            node.right, deleted = self._delete(node.right, score)
            return node, deleted


        if node.left is None and node.right is None:
            return None, True


        if node.left is None:
            return node.right, True

        if node.right is None:
            return node.left, True


        successor = self._find_min(node.right)

        node.score = successor.score
        node.params = successor.params

        node.right, _ = self._delete(node.right, successor.score)

        return node, True

    def _search(self, node, score):
        if node is None:
            return None
        
        if score == node.score :
            return node
        
        if score < node.score:
            return self._search(node.left, score)
        
        return self._search(node.right, score)
        

    def _find_min(self, node):
        if node is None:
            return None
        
        if node.left is None:
            return node
        return self._find_min(node.left)

    def _height(self, node):
        if node is None:
            return 0
        
        left_height = self._height(node.left)
        right_height = self._height(node.right)

        return 1 + max(left_height, right_height)



    def _check_balanced(self, node):
        """
        Returns the height of the subtree if balanced, or -1 if not.

        A node is unbalanced if |left_height - right_height| > 1.
        """
        if node is None:
            return 0
        
        left_height = self._check_balanced(node.left)
        if left_height == -1:
            return -1
        
        right_height = self._check_balanced(node.right)
        if right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return 1 + max(left_height, right_height)

    def _inorder(self, node, result):
        if node is None:
            return
        
        self._inorder(node.left, result)
        result.append(node)
        self._inorder(node.right, result)
        

    def _preorder(self, node, result):
        if node is None:
            return

        result.append(node)
        self._preorder(node.left, result)
        self._preorder(node.right, result)

    def _postorder(self, node, result):
        if node is None:
            return

        self._postorder(node.left, result)
        self._postorder(node.right, result)
        result.append(node)