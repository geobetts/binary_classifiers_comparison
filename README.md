# binary_classifiers_comparison

![](/alexander-sinn-YYUM2sNvnvU-unsplash.jpg)

Automated comparison of scikit-learn binary classifiers.

## Brief description of GridSearchClassifier

GridSearchClassifier is a tool to find the best performing classifier (according to accuracy/F1, train time and test/deployment time) 
using a given labelled training set and labelled test set. The user can input a list of scikit-learn classifiers and classifiers compatible
with scikit-learn. The tool searches through classifier provided in the list. The tool can be used both to choose between different algorithms
and to choose hyperparameters for a specific algorithm.

## Recommended usage

- On smaller data frames.
- Initial exlploratory of different algorithm performance. 
- To rule out algorithms.
- Hyperparamter selection.

## Examples

There are examples of using GridSearchClassifier on face recognition data in /example_binary_face_recognition.py
