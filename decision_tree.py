import numpy as np
from operations import find_mode


class Node():
    def __init__(self, return_value=None, split_value=None, attribute_name=None, attribute_index=None, branches=None):
        if branches is None:
            branches = []
        self.branches = branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.split_value = split_value
        self.return_value = return_value


class DecisionTree():
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def visualize(self, branch=None, level=0):
        if branch is None:
            branch = self.tree
        self._visualize_helper(branch, level)

        if len(branch.branches) > 0:
            left, right = branch.branches
            if left is not None:
                self.visualize(left, level + 1)

            if left is not None and right is not None:
                tab_level = "  " * level
                print(f"{level}: {tab_level} else:")

            if right is not None:
                self.visualize(right, level + 1)

    def _visualize_helper(self, tree, level):
        tab_level = "  " * level
        if len(tree.branches) == 0:
            print(f"{level}: {tab_level} Predict {tree.return_value}")
        elif len(tree.branches) == 2:
            print(f"{level}: {tab_level} if {tree.attribute_name} <= {tree.split_value:.1f}:")

    def fit(self, features, labels):
        self._check_input(features)

        self.tree = self._create_tree(
            features=features,
            labels=labels,
            used_attributes=[],
            default=0,
        )

    def _create_tree(self, features, labels, used_attributes, default):
        if not len(features) and not len(labels):
            return Node(default)

        if len(np.unique(labels)) == 1:
            return Node(labels[0][0])
    
        info_gain = -1
        best_attribute = None
        for i in range(len(self.attribute_names)):
            if i not in used_attributes:
                temp_info_gain = information_gain(features, i, labels)
                if temp_info_gain > info_gain:
                    info_gain = temp_info_gain
                    best_attribute = i

        attribute_values = features[:, best_attribute]
            
        if info_gain == -1:
            return Node(find_mode(labels))

        new_used_attributes = used_attributes + [best_attribute]
        split = 0
        temp = np.unique(attribute_values)
        if len(temp) == 2 and temp[0] == 0 and temp[1] == 1:
            split = 0.5
        else:
            split = np.median(attribute_values)
        left_mask = np.where(attribute_values <= split)
        right_mask = np.where(attribute_values > split)
        return Node(None, 
                    split, 
                    self.attribute_names[best_attribute], 
                    best_attribute,
                    [self._create_tree(features[left_mask], labels[left_mask], new_used_attributes, find_mode(labels)), 
                     self._create_tree(features[right_mask], labels[right_mask], new_used_attributes, find_mode(labels))])
            


    def predict(self, features):
        self._check_input(features)
        def traverse_tree(tree, attr_values):
            # if return value is not None, append prediction
            # else traverse the tree
            #   use branches
            #   decide to move to left or right node
            if tree.return_value is not None:
                return tree.return_value
            if attr_values[tree.attribute_index] <= tree.split_value:
                return traverse_tree(tree.branches[0], attr_values)
            return traverse_tree(tree.branches[1], attr_values)

        arr = []
        for example in features:
            arr.append(traverse_tree(self.tree, example)) # append prediction from decision tree
        return np.array(arr).reshape(-1, 1)

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    H_S = 0.0
    for c in counts:
        p_c = c / np.sum(counts)
        H_S -= p_c * np.log2(p_c)

    return H_S

def information_gain(features, attribute_index, labels):
    H_S = entropy(labels)
    attr_values = features[:, attribute_index]
    ds1, ds2 = None, None
    if len(np.unique(attr_values)) == 2:
        ds1 = labels[attr_values == 0]
        ds2 = labels[attr_values == 1] 
    else:
        m = np.median(attr_values)
        ds1 = labels[attr_values <= m]
        ds2 = labels[attr_values > m]
    H_SA = (len(ds1)/len(labels))*entropy(ds1) + (len(ds2)/len(labels))*entropy(ds2)
    return H_S - H_SA
    

if __name__ == '__main__':
    # Manually construct a simple decision tree and visualize it
    attribute_names = ['Outlook', 'Temp']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    root = Node(
        attribute_name="Outlook", attribute_index=0,
        split_value=0.5, branches=[])

    left = Node(
        attribute_name="Temp", attribute_index=1,
        split_value=0.5, branches=[])

    left_left = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left_right = Node(
        attribute_name=None, attribute_index=None,
        return_value=0, branches=[])

    right = Node(
        attribute_name=None, attribute_index=None,
        return_value=1, branches=[])

    left.branches = [left_left, left_right]
    root.branches = [left, right]
    decision_tree.tree = root

    decision_tree.visualize()

    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print(DecisionTree().predict(features))
