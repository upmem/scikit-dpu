specs: parameters of the dataset
X: features[point index][feature index]
y: targets[point index]
minmaxes: [min | max][feature index]
counts: separate splits to evaluate
    split_<feature_index>_instruction: feature and threshold to evaluate a split on
    split_<feature_index>_nodecounts: <point count in left node> <point count in right node>
    split_<feature_index>_gini: point count per class [class id] [left node | right node]
commits: commits to do in succession
    initial_leaf_indices: all 0
    commit_<commit index>_instruction: leaf, feature and threshold to commit the split on
    commit_<commit index>_leaf_indices: resulting array of leaf indices after commit <commit index>