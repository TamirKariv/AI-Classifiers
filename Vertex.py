# a vertex in the decision tree contains the fields value
# or the label if it's type is a leaf.
class Vertex(object):
    def __init__(self, field=None, successors=None, label=None, type=None, level=None, val=None):
        self.field = field
        self.successors = successors
        self.label = label
        self.type = type
        self.level = level
        self.val = val
