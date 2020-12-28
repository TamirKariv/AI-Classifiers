from Classifiers import Tree
from Classifiers import KNN
from Classifiers import NaiveBayes

# read the whole data from the file
def read_from_file(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in range(len(content)):
        content[i] = content[i].split("\t")
    return content

# extract the inputs from the file's data
def get_inputs(train_file, test_file):
    train = read_from_file(train_file)
    test = read_from_file(test_file)
    fields = train[0]
    labels = []
    del fields[-1]
    del train[0]
    del test[0]
    possible_vals = {}
    for row in test:
        labels.append(row[-1])
    for f in fields: possible_vals[f] = set()
    for row in train:
        for j in range(len(row) - 1):
            possible_vals[fields[j]].add(row[j])
    return train, test, fields, possible_vals, labels


def main():
    # get the input from the files.
    train, test, fields, possible_vals, labels = get_inputs("train.txt", "test.txt")
    # get the results from the decision tree model.
    tree = Tree(train, test, fields, possible_vals)
    tree.vertex_root = tree.DTL(0, train, fields, tree.get_majority_of_labels(train))
    tree_str = tree.convert_to_string(tree.vertex_root)
    tree_results = tree.get_results()
    tree_acc = check_acc(tree_results, labels)
    # get the results from the knn model.
    knn = KNN(train, test, 5)
    knn_results = knn.get_results()
    knn_acc = check_acc(knn_results, labels)
    # get the results from the naive bayes model.
    nb = NaiveBayes(train, test, fields, possible_vals)
    nb_results = nb.get_results()
    nb_acc = check_acc(nb_results, labels)
    # write the results to the file.
    write_to_file("output.txt", tree_str, tree_acc, knn_acc, nb_acc)

# write the tree and the accuracies to the file.
def write_to_file(file_name, tree_str, tree_acc, knn_acc, nb_acc):
    f = open(file_name, "w")
    f.write(tree_str)
    f.write("\n")
    f.write("{}\t{}\t{}".format(tree_acc, knn_acc, nb_acc))
    f.close()

# check the accuracy of the model.
def check_acc(predictions, labels):
    error = 0
    for prediction, label in zip(predictions, labels):
        if prediction != label:
            error += 1
    return round(1 - float(error / len(predictions)), 2)


if __name__ == '__main__':
    main()
