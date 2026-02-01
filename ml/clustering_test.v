module ml

fn test__kmeans() {
	// Simple 2D data with 2 clear clusters
	data := [
		[1.0, 1.0],
		[1.5, 1.5],
		[10.0, 10.0],
		[10.5, 10.5],
	]
	
	model := kmeans(data, 2, 100)
	
	// Should have 2 centroids
	assert model.centroids.len == 2
	// Should have 4 labels
	assert model.labels.len == 4
	// Labels should be 0 or 1
	for label in model.labels {
		assert label == 0 || label == 1
	}
}

fn test__kmeans_single_cluster() {
	// All points very close together
	data := [
		[1.0, 1.0],
		[1.01, 1.01],
		[1.02, 1.02],
	]
	
	model := kmeans(data, 1, 100)
	
	assert model.centroids.len == 1
	assert model.labels.len == 3
	// All points should be in same cluster
	for label in model.labels {
		assert label == 0
	}
}

fn test__kmeans_predict() {
	data := [
		[1.0, 1.0],
		[1.5, 1.5],
		[10.0, 10.0],
		[10.5, 10.5],
	]
	
	model := kmeans(data, 2, 100)
	
	// Predict on new point
	new_point := [1.2, 1.3]
	label := kmeans_predict(model, [new_point])
	
	// Should have same cluster as first point
	assert label[0] == model.labels[0]
}

fn test__kmeans_inertia() {
	data := [
		[1.0, 1.0],
		[1.5, 1.5],
		[10.0, 10.0],
		[10.5, 10.5],
	]
	
	model := kmeans(data, 2, 100)
	inertia := kmeans_inertia(model, data)
	
	// Inertia should be non-negative
	assert inertia >= 0.0
}

fn test__silhouette_coefficient() {
	data := [
		[1.0, 1.0],
		[1.1, 1.1],
		[10.0, 10.0],
		[10.1, 10.1],
	]
	
	labels := [0, 0, 1, 1]
	silhouette := silhouette_coefficient(data, labels)
	
	// Silhouette should be between -1 and 1
	assert silhouette >= -1.0 && silhouette <= 1.0
	// Good clustering should have high silhouette
	assert silhouette > 0.5
}

fn test__hierarchical_clustering() {
	data := [
		[1.0, 1.0],
		[1.5, 1.5],
		[10.0, 10.0],
		[10.5, 10.5],
	]
	
	result := hierarchical_clustering(data, 2)
	
	// Should have 2 clusters
	assert result.clusters.len == 2
	// Total points should equal input
	mut total_points := 0
	for cluster in result.clusters {
		total_points += cluster.len
	}
	assert total_points == 4
}

fn test__dbscan() {
	data := [
		[1.0, 1.0],
		[1.1, 1.1],
		[10.0, 10.0],
		[10.1, 10.1],
	]
	
	labels := dbscan(data, 0.5, 2)
	
	// Should have labels for all points
	assert labels.len == 4
	// Labels should be >= 0 (0 = noise, >0 = cluster)
	for label in labels {
		assert label >= 0
	}
}

fn test__dbscan_no_clusters() {
	// Points too far apart to cluster
	data := [
		[0.0, 0.0],
		[100.0, 100.0],
		[200.0, 200.0],
	]
	
	labels := dbscan(data, 1.0, 2)
	
	assert labels.len == 3
}
