from bst_toolkit.bst import BST

tree = BST()

tree.insert(10, {"a": 10})
tree.insert(5, {"a": 5})
tree.insert(15, {"a": 15})
tree.insert(3, {"a": 3})
tree.insert(7, {"a": 7})

print("Length:", len(tree))
print("Min:", tree.find_min())
print("Max:", tree.find_max())
print("Search 7:", tree.search(7))
print("Search 99:", tree.search(99))
print("Height:", tree.height())
print("Balanced:", tree.is_balanced())
print("Inorder:", tree.inorder())
print("Preorder:", tree.preorder())
print("Postorder:", tree.postorder())
print("Level order:", tree.level_order())

tree.delete(5)
print("After delete 5:", tree.inorder())
print("Length:", len(tree))