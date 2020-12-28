import math
from Vertex import Vertex


# The Decision Tree classifier - classify the input using a tree where each vertex is a field value
# the leaves of the tree contain the labels.
class Tree(object):
    def __init__(self, train_x, test_x, fields, fields_values):
        self.train_x = train_x
        self.test_x = test_x
        self.fields = fields
        self.fields_values = fields_values
        self.vertex_root = None
        self.label = self.get_majority_of_labels(self.train_x)

    # get the possible field values.
    def get_field_val(self, field):
        return sorted(list(self.fields_values[field]))

    # get the label which is most common
    def get_majority_of_labels(self, train_x):
        positive_labels = 0
        false_labels = 0
        for x in train_x:
            if x[-1] == "yes":
                positive_labels += 1
            else:
                false_labels += 1
        return "yes" if positive_labels > false_labels else "no"

    # calculate the entropy
    def calculate_entropy(self, train_X):
        if not train_X:
            return 0
        positive_labels = 0
        for x in train_X:
            if x[-1] == "yes":
                positive_labels += 1
        positive_labels /= len(train_X)
        if positive_labels == 1:
            return 0
        if positive_labels == 0:
            return 0
        negative_labels = 1 - positive_labels
        log_positive = -positive_labels * math.log(positive_labels, 2)
        log_negative = -negative_labels * math.log(negative_labels, 2)
        return log_positive + log_negative

    # split the tree by a field value
    def split_by_field_val(self, field, field_val, train_x):
        split = []
        for x in train_x:
            if x[self.fields.index(field)] == field_val:
                split.append(x)
        return split

    # calculate the information gain
    def caclculate_information_gain(self, train_x, field):
        info_gain = 0
        total_size = float(len(train_x))
        for v in self.get_field_val(field):
            split = self.split_by_field_val(field, v, train_x)
            split_size = float(len(split))
            share = split_size / total_size
            entropy = self.calculate_entropy(split)
            info_gain = info_gain + share * entropy
        return info_gain

    # pick the field with the maximum information gain to split the tree by.
    def choose_next_field(self, train_x, fields):
        max_info_gain = -9999
        for f in fields:
            cur_info_gain = self.calculate_entropy(train_x) - self.caclculate_information_gain(train_x, f)
            if cur_info_gain > max_info_gain:
                max_info_gain = cur_info_gain
                next_field = f
        return next_field

    # build the tress using DTL algorithm.
    def DTL(self, current_level, train_x, fields, label):
        if not train_x:
            leaf = Vertex(None, None, label, "Leaf", current_level)
            return leaf
        elif not self.labels_are_different(train_x):
            leaf_label = train_x[-1][-1]
            leaf = Vertex(None, None, leaf_label, "Leaf", current_level)
            return leaf
        elif not fields:
            positive_labels = 0
            false_labels = 0
            for x in train_x:
                if x[-1] == "yes":
                    positive_labels += 1
                else:
                    false_labels += 1
            leaf_label = "yes" if positive_labels > false_labels else "no"
            leaf = Vertex(None, None, leaf_label, "Leaf", current_level)
            return leaf
        else:
            successors = []
            chosen_field = self.choose_next_field(train_x, fields)
            successor_fields = fields[:]
            successor_fields.remove(chosen_field)
            vertex_root = Vertex(chosen_field, successors, None, "Root", current_level)
            for v in self.get_field_val(chosen_field):
                split = self.split_by_field_val(chosen_field, v, train_x)
                subtree = self.DTL(current_level + 1, split, successor_fields, label)
                subtree.val = v
                successors.append(subtree)
        return vertex_root

    # check if there are different labels in the examples
    def labels_are_different(self, train_x):
        cur_label = None
        first_label = True
        for x in train_x:
            if first_label:
                cur_label = x[-1]
                first_label = False
            else:
                if cur_label != x[-1]:
                    return True
                cur_label = x[-1]
        return False

    # convert the tree to a string form
    def convert_to_string(self, vertex):
        tree = ""
        for successor in sorted(vertex.successors, key=lambda ver: ver.val):
            tabs = vertex.level * "\t"
            tree = tree + tabs
            if vertex.level >= 1:
                tree = tree + "|"
            val = successor.val
            tree += vertex.field + "=" + val
            if successor.type == "Leaf":
                leaf_label = successor.label
                tree += ":" + leaf_label + "\n"
            else:
                tree += "\n" + self.convert_to_string(successor)
        return tree

    # get a label for a given input
    def get_label(self, x, vertex):
        for successor in vertex.successors:
            if x[self.fields.index(vertex.field)] == successor.val:
                if successor.type == "Leaf":
                    return successor.label
                else:
                    return self.get_label(x, successor)
        positive_labels = 0
        negative_labels = 0
        for successor in vertex.successors:
            if successor.type == "Leaf":
                if successor.label == "yes":
                    positive_labels += 1
                else:
                    negative_labels += 1
        return "yes" if positive_labels > negative_labels else "no"

    # get the predictions of the tree for the inputs.
    def get_results(self):
        results = []
        for x in self.test_x:
            label = self.get_label(x, self.vertex_root)
            results.append(label)
        return results


# The KNN Classifier - classify the input by calculating the hamming distance of an input from all the examples.
# from the k examples with the shortest distance take the labels and pick the most occurring one.
class KNN:
    def __init__(self, train_x, test_x, num_of_neighbours):
        self.train_x = train_x
        self.test_x = test_x
        self.num_of_neighbours = num_of_neighbours

    # calculate the hamming_distance.
    def calculate_distance(self, x1, x2):
        hamming_distance = 0
        for f1, f2 in zip(x1[:-1], x2[:-1]):
            if f1 != f2:
                hamming_distance += 1
        return hamming_distance

    # get a label for a given input
    def get_label(self, input):
        labels = []
        distances = []
        for x in self.train_x:
            dist = self.calculate_distance(x, input)
            distances.append(dist)
            labels.append(x[-1])
        sorted_labels = [label for _, label in sorted(zip(distances, labels))][:self.num_of_neighbours]
        positive_labels = 0
        negative_labels = 0
        for label in sorted_labels:
            if label == "yes":
                positive_labels += 1
            else:
                negative_labels += 1
        return "yes" if positive_labels > negative_labels else "no"

    # get the predictions.
    def get_results(self):
        results = []
        for x in self.test_x:
            label = self.get_label(x)
            results.append(label)
        return results

# the Naive Bayes classifier - for a given example calculate the probability of each label
# by the assumption of independence between the fields, then pick the label with the higher probability.
class NaiveBayes:
    def __init__(self, train_x, test_x, fields, fields_values):
        self.train_x = train_x
        self.test_x = test_x
        self.fields = fields
        self.num_of_possible_values = self.get_length_of_fields(fields_values)
        self.positive_examples, self.positive_prob = self.split_by_labels("yes")
        self.negative_examples, self.negative_prob = self.split_by_labels("no")
        self.num_of_positive_examples = len(self.positive_examples)
        self.num_of_negative_examples = len(self.negative_examples)

    # split the examples by their labels also calculate the label's probability by it's occurrence.
    def split_by_labels(self, label):
        split_examples = []
        for x in self.train_x:
            if x[-1] == label:
                split_examples.append(x)
        prob = float(len(split_examples) / len(self.train_x))
        return split_examples, prob

    # get the length of all the fields by their possible values.
    def get_length_of_fields(self, fields_values):
        sizes = []
        for field in self.fields:
            sizes.append(len(fields_values[field]))
        return sizes

    # get the probability of each field.
    def get_probabilities(self, input):
        probabilities = {}
        for field in self.fields:
            idx = self.fields.index(field)
            field_counter = [0, 0]
            for x in self.positive_examples:
                if x[idx] == input[idx]:
                    field_counter[0] += 1
            sum = self.num_of_positive_examples + self.num_of_possible_values[idx]
            positive_prob = float(field_counter[0] / sum)
            for x in self.negative_examples:
                if x[idx] == input[idx]:
                    field_counter[1] += 1
            sum = self.num_of_negative_examples + self.num_of_possible_values[idx]
            negative_prob = float(field_counter[1] / sum)
            probabilities[field] = positive_prob, negative_prob
        return probabilities

    # get a label for a given input.
    def get_label(self, input):
        probabilities = self.get_probabilities(input)
        prob_vals = probabilities.values()
        positive_probs = 1
        negative_probs = 1
        for prob in prob_vals:
            positive_probs *= prob[0]
            negative_probs *= prob[1]
        positive_probs *= self.positive_prob
        negative_probs *= self.negative_prob
        return "yes" if positive_probs > negative_probs else "no"

    # get the predictions.
    def get_results(self):
        res = []
        for x in self.test_x:
            label = self.get_label(x)
            res.append(label)
        return res
