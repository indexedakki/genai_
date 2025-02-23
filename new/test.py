class Node():
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class solution:
    def delete_middle_node(self, head: Node):
        if not head or not head.next:
            return None
        slow = head
        fast = head
        prev = None

        while fast and fast.next:
            fast = fast.next.next
            prev = slow
            slow = slow.next

        prev.next = slow.next
        return head

head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
obj = solution()
obj.delete_middle_node(head)
# root=TreeNode(3)
# root.right=TreeNode(20)
# root.left=TreeNode(9)
# root.right.left=TreeNode(15)
# root.right.right=TreeNode(7)
# obj = solution()
# obj.maxDepth(root)
