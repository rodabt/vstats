import ml
import utils

fn test__random_forest_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, 'Failed to load Titanic dataset: ${err}'
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := ml.random_forest_classifier(x_train, y_train, 10, 8)
	assert model.trained == true, 'model should be trained'
	assert model.num_trees == 10, 'should have 10 trees'
	assert model.trees.len == 10, 'trees array should have 10 elements'
	
	predictions := ml.random_forest_classifier_predict(model, x_test)
	assert predictions.len == x_test.len, 'predictions should match test set size'
	
	for pred in predictions {
		assert pred == 0 || pred == 1, 'predictions should be 0 or 1'
	}
	
	acc := ml.accuracy(y_test, predictions)
	prec := ml.precision(y_test, predictions, 1)
	rec := ml.recall(y_test, predictions, 1)
	f1 := ml.f1_score(y_test, predictions, 1)
	
	println('\n=== Random Forest on Titanic ===')
	println('Test set size: ${x_test.len}')
	println('Accuracy:  ${acc:.4f} (expected: 0.65-0.75)')
	println('Precision: ${prec:.4f}')
	println('Recall:    ${rec:.4f}')
	println('F1 Score:  ${f1:.4f}')
	
	assert acc >= 0.65, 'Random Forest should achieve 65%+ accuracy'
	assert prec >= 0.0, 'Precision should be valid'
	assert rec >= 0.0, 'Recall should be valid'
	assert f1 >= 0.0, 'F1 should be valid'
}
