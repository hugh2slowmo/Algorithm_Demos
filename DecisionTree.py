from collections import defaultdict

# some utils function
def unique_values(dataset, feature_col):
        return set([data[feature_col] for data in dataset])

def class_counts(dataset):
    counts = defaultdict(int)
    for data in dataset:
        label = data[-1]
        counts[label] += 1
    return counts
        
def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)


# Node and Leaf for Decision Tree
class Node:
    def __init__(self, assertion, branch_T, branch_F):
        self.assertion = assertion
        self.branch_True = branch_T
        self.branch_False = branch_F

class Leaf:
    def __init__(self, dataset):
        counts = class_counts(dataset)
        total = sum(counts.values()) * 1.0
        prob = {}
        for lbl in counts:
            prob[lbl] = f'{int(counts[lbl] / total * 100)} %'
        self.predictions = prob



class Assertion:
    def __init__(self, feature_col, feature_val, feature_name):
        self.feature_column = feature_col
        self.feature_value = feature_val
        self.feature_name = feature_name
    
    def split(self, dataset):
        branch_True = []
        branch_False = []
        for data in dataset:
            val = data[self.feature_column]
            if is_numeric(val):
                if val >= self.feature_value:
                    branch_True.append(data)
                else:
                    branch_False.append(data)
            else:
                if val == self.feature_value:
                    branch_True.append(data)
                else:
                    branch_False.append(data)
        return branch_True, branch_False

    def match(self, data):
        val = data[self.feature_column]
        if is_numeric(val):
            return val > self.feature_value
        else:
            return val == self.feature_value
    
    def __repr__(self):
        c = '=='
        if is_numeric(self.feature_value):
            c = '>'
        return f'{self.feature_name} {c} {self.feature_value}?'



class DecisionTreeClassifier:
    def __init__(self):
        self.data_header = None
        self.tree = None

    def fit(self, data_header, dataset):
        self.data_header = data_header
        self.tree = self.build(dataset)
        self.print_tree(self.tree)
    
    def predict(self, data, node=None):
        if not node:
            node = self.tree
        if isinstance(node, Leaf):
            return node.predictions
        elif node.assertion.match(data):
            return self.predict(data, node.branch_True)
        else:
            return self.predict(data, node.branch_False)

     
    def gini_impurity(self, dataset):
        # If it confuse you, just treat it as the loss function of Decision Tree
        impurity = 1
        counts = class_counts(dataset)
        for lbl in counts:
            impurity -= (counts[lbl] / float(len(dataset))) ** 2
        return impurity

    def gain_caculate(self, left, right, node_impurity):
        p = len(left) / float(len(left) + len(right))
        return node_impurity - (p * self.gini_impurity(left) + (1 - p) * self.gini_impurity(right))

    def get_best_split(self, dataset):
        best_gain = 0
        best_assertion = None
        node_impurity = self.gini_impurity(dataset)
        n_features = len(dataset[0]) - 1

        for feature_col in range(n_features):
            all_values = unique_values(dataset, feature_col)
            feature_name = self.data_header[feature_col]
            for val in all_values:
                assertion = Assertion(feature_col, val, feature_name)
                branch_T, branch_F = assertion.split(dataset)
                if len(branch_T) == 0 or len(branch_F) == 0: continue
                gain = self.gain_caculate(branch_T, branch_F, node_impurity)
                if gain > best_gain:
                    best_gain, best_assertion = gain, assertion
        
        return best_gain, best_assertion

    def build(self, dataset):
        gain, assertion = self.get_best_split(dataset)
        if gain == 0:
            return Leaf(dataset)
        
        branch_T, branch_F = assertion.split(dataset)
        node_T = self.build(branch_T)
        node_F = self.build(branch_F)
        return Node(assertion, node_T, node_F)
    
    def print_tree(self, node, spacing=''):
        if isinstance(node, Leaf):
            print(f'{spacing}Prediction', node.predictions)
            return
        print(f'{spacing}{node.assertion}')
        print(f'{spacing}--> True:')
        self.print_tree(node.branch_True, '   ')
        print(f'{spacing}--> False:')
        self.print_tree(node.branch_False, '   ')



if __name__ == '__main__':

    training_data = [
        ['Green', 3, 'Sweet', 'Apple'],
        ['Yellow', 3, 'Sweet', 'Apple'],
        ['Red', 1, 'Sweet', 'Grape'],
        ['Red', 1, 'Sour', 'Grape'],
        ['Yellow', 3, 'Sour', 'Lemon'],
    ]

    header = ['color', 'size', 'taste',"label"]

    clf = DecisionTreeClassifier()
    clf.fit(header, training_data)

    testing_data = [
            ['Green', 3, 'Sweet', 'Apple'],
            ['Yellow', 4, 'Sweet', 'Apple'],
            ['Red', 2, 'Sour', 'Grape'],
            ['Red', 1, 'Sweet', 'Grape'],
            ['Yellow', 3, 'Sour', 'Lemon'],
        ]

    for i in testing_data:
        res = clf.predict(i)
        print(f'Ture label is: {i[-1]}, Prediction is {res}')