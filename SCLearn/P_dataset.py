from sklearn.datasets import load_iris

iris_data = load_iris()
keys = iris_data.keys()
print('IRIS''s Keys', keys)

print('feature names''s shape:', len(iris_data.feature_names))
print('feature names :', iris_data.feature_names)

print('target names''s shape:', len(iris_data.target_names))
print('target names :', iris_data.target_names)

print('data shape :', iris_data.data.shape)
print('data :', iris_data['data'])

print('target shape :', iris_data.target.shape)
print('target :', iris_data.target)

print('iris frame: ', iris_data.frame)
print('iris DESCRIPTION: ', iris_data.DESCR)
