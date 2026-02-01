module ml

import linalg
import math

pub struct KMeansModel {
	centroids [][]f64
	labels    []int
	iterations int
}

pub struct HierarchicalClustering {
	clusters [][]int
	distances []f64
}

// K-Means Clustering Algorithm
pub fn kmeans(data [][]f64, k int, max_iterations int) KMeansModel {
	assert data.len > 0, "data cannot be empty"
	assert k > 0 && k <= data.len, "k must be between 1 and number of samples"
	
	n := data.len
	d := data[0].len
	
	// Random initialization of centroids
	mut centroids := [][]f64{len: k}
	for i in 0 .. k {
		centroids[i] = data[i % n].clone()
	}
	
	mut labels := []int{len: n}
	mut prev_centroids := [][]f64{len: k}
	for i in 0 .. k {
		prev_centroids[i] = centroids[i].clone()
	}
	
	// Iterate
	for iteration in 0 .. max_iterations {
		// Assign points to nearest centroid
		for i in 0 .. n {
			mut min_dist := f64(1e10)
			mut nearest_centroid := 0
			
			for j in 0 .. k {
				dist := linalg.distance(data[i], centroids[j])
				if dist < min_dist {
					min_dist = dist
					nearest_centroid = j
				}
			}
			
			labels[i] = nearest_centroid
		}
		
		// Update centroids
		for j in 0 .. k {
			mut cluster_points := [][]f64{}
			for i in 0 .. n {
				if labels[i] == j {
					cluster_points << data[i].clone()
				}
			}
			
			if cluster_points.len > 0 {
				mut new_centroid := []f64{len: d, init: 0.0}
				for point in cluster_points {
					for d_idx in 0 .. d {
						new_centroid[d_idx] += point[d_idx]
					}
				}
				for d_idx in 0 .. d {
					new_centroid[d_idx] /= f64(cluster_points.len)
				}
				centroids[j] = new_centroid
			}
		}
		
		// Check convergence
		mut converged := true
		for j in 0 .. k {
			if linalg.distance(centroids[j], prev_centroids[j]) > 1e-6 {
				converged = false
				break
			}
		}
		
		if converged {
			return KMeansModel{
				centroids: centroids
				labels: labels
				iterations: iteration + 1
			}
		}
		
		for j in 0 .. k {
			prev_centroids[j] = centroids[j].clone()
		}
	}
	
	return KMeansModel{
		centroids: centroids
		labels: labels
		iterations: max_iterations
	}
}

// Predict cluster for new points
pub fn kmeans_predict(model KMeansModel, data [][]f64) []int {
	mut predictions := []int{len: data.len}
	
	for i in 0 .. data.len {
		mut min_dist := f64(1e10)
		mut nearest := 0
		
		for j in 0 .. model.centroids.len {
			dist := linalg.distance(data[i], model.centroids[j])
			if dist < min_dist {
				min_dist = dist
				nearest = j
			}
		}
		
		predictions[i] = nearest
	}
	
	return predictions
}

// Inertia: sum of squared distances to nearest centroid
pub fn kmeans_inertia(model KMeansModel, data [][]f64) f64 {
	mut inertia := 0.0
	
	for i in 0 .. data.len {
		cluster := model.labels[i]
		dist := linalg.distance(data[i], model.centroids[cluster])
		inertia += dist * dist
	}
	
	return inertia
}

// Silhouette coefficient (measure of cluster quality)
pub fn silhouette_coefficient(data [][]f64, labels []int) f64 {
	assert data.len == labels.len, "data and labels must have same length"
	
	n := data.len
	mut silhouette_scores := []f64{len: n}
	
	for i in 0 .. n {
		label := labels[i]
		
		// Calculate average distance to points in same cluster
		mut a := 0.0
		mut same_cluster_count := 0
		for j in 0 .. n {
			if i != j && labels[j] == label {
				a += linalg.distance(data[i], data[j])
				same_cluster_count++
			}
		}
		if same_cluster_count > 0 {
			a /= f64(same_cluster_count)
		}
		
		// Calculate minimum average distance to other clusters
		mut b := f64(1e10)
		for other_label in 0 .. 10 {
			if other_label != label {
				mut dist_sum := 0.0
				mut other_count := 0
				for j in 0 .. n {
					if labels[j] == other_label {
						dist_sum += linalg.distance(data[i], data[j])
						other_count++
					}
				}
				if other_count > 0 {
					b = math.min(b, dist_sum / f64(other_count))
				}
			}
		}
		
		// Calculate silhouette score
		max_val := math.max(a, b)
		if max_val > 0 {
			silhouette_scores[i] = (b - a) / max_val
		} else {
			silhouette_scores[i] = 0
		}
	}
	
	// Average silhouette score
	mut avg := 0.0
	for score in silhouette_scores {
		avg += score
	}
	return avg / f64(n)
}

// Hierarchical Clustering - Single Linkage
pub fn hierarchical_clustering(data [][]f64, num_clusters int) HierarchicalClustering {
	assert data.len > 0, "data cannot be empty"
	assert num_clusters > 0 && num_clusters <= data.len, "invalid number of clusters"
	
	n := data.len
	
	// Initialize: each point is its own cluster
	mut clusters := [][]int{len: n}
	for i in 0 .. n {
		clusters[i] = [i]
	}
	
	mut distances := []f64{}
	
	// Merge until desired number of clusters
	for {
		if clusters.len <= num_clusters {
			break
		}
		
		// Find closest pair of clusters
		mut min_dist := f64(1e10)
		mut merge_i := 0
		mut merge_j := 1
		
		for i in 0 .. clusters.len {
			for j in (i + 1) .. clusters.len {
				// Single linkage: minimum distance between any two points
				mut dist := f64(1e10)
				for pi in clusters[i] {
					for pj in clusters[j] {
						d := linalg.distance(data[pi], data[pj])
						if d < dist {
							dist = d
						}
					}
				}
				
				if dist < min_dist {
					min_dist = dist
					merge_i = i
					merge_j = j
				}
			}
		}
		
		// Merge clusters
		mut new_cluster := clusters[merge_i].clone()
		new_cluster << clusters[merge_j]
		
		// Remove old clusters and add merged one
		mut new_clusters := [][]int{}
		for i in 0 .. clusters.len {
			if i != merge_i && i != merge_j {
				new_clusters << clusters[i]
			}
		}
		new_clusters << new_cluster
		
		clusters = new_clusters.clone()
		distances << min_dist
	}
	
	return HierarchicalClustering{
		clusters: clusters
		distances: distances
	}
}

// DBSCAN Clustering (density-based)
pub fn dbscan(data [][]f64, eps f64, min_points int) []int {
	assert data.len > 0, "data cannot be empty"
	
	n := data.len
	mut labels := []int{len: n, init: -1}
	mut cluster_id := 0
	
	for i in 0 .. n {
		if labels[i] != -1 {
			continue
		}
		
		// Find neighbors
		mut neighbors := []int{}
		for j in 0 .. n {
			if linalg.distance(data[i], data[j]) <= eps {
				neighbors << j
			}
		}
		
		// Point is noise
		if neighbors.len < min_points {
			labels[i] = 0
			continue
		}
		
		// Start new cluster
		labels[i] = cluster_id
		
		// Process neighbors
		mut queue := neighbors.clone()
		for queue.len > 0 {
			point := queue.pop()
			
			if labels[point] == 0 {
				labels[point] = cluster_id
			}
			
			if labels[point] != -1 {
				continue
			}
			
			labels[point] = cluster_id
			
			// Find neighbors of neighbor
			mut point_neighbors := []int{}
			for j in 0 .. n {
				if linalg.distance(data[point], data[j]) <= eps {
					point_neighbors << j
				}
			}
			
			if point_neighbors.len >= min_points {
				for neighbor in point_neighbors {
					if labels[neighbor] == -1 {
						queue << neighbor
					}
				}
			}
		}
		
		cluster_id++
	}
	
	return labels
}
