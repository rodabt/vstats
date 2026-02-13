module utils

// ============================================================================
// Phase 3: Parallel Batch Processing
// Hardware acceleration through multi-threading
// Uses V's built-in thread pool and channel support
// ============================================================================

// ParallelConfig - Configuration for parallel execution
pub struct ParallelConfig {
pub mut:
	num_workers int  // Number of parallel workers (0 = auto-detect)
	chunk_size  int  // Minimum chunk size per worker
}

// default_parallel_config - Create default parallel configuration
// Returns a config with sensible defaults (4 workers)
pub fn default_parallel_config() ParallelConfig {
	return ParallelConfig{
		num_workers: 4
		chunk_size:  1
	}
}

// parallel_for - Execute a function in parallel over a range using channels
// Divides the range [start, end) among workers and calls callback for each index
pub fn parallel_for(start int, end int, config ParallelConfig, callback fn(int)) {
	if end <= start {
		return
	}

	num_workers := if config.num_workers > 0 { config.num_workers } else { 4 }
	total_items := end - start

	// If not enough items for parallelization, run sequentially
	if total_items < config.chunk_size * num_workers || num_workers == 1 {
		for i in start .. end {
			callback(i)
		}
		return
	}

	// Calculate items per worker
	items_per_worker := total_items / num_workers
	remainder := total_items % num_workers

	// Channel for worker coordination
	ch := chan int{cap: num_workers}

	// Launch workers
	mut worker_start := start
	for w in 0 .. num_workers {
		// Distribute remainder among first workers
		worker_items := items_per_worker + if w < remainder { 1 } else { 0 }
		worker_end := worker_start + worker_items

		// Launch goroutine for this worker's range
		go fn (w_start int, w_end int, cb fn(int), done chan int) {
			for i in w_start .. w_end {
				cb(i)
			}
			done <- 1  // Signal completion
		}(worker_start, worker_end, callback, ch)

		worker_start = worker_end
	}

	// Wait for all workers to complete
	for _ in 0 .. num_workers {
		_ = <-ch
	}
}

// parallel_map - Map a function over a slice in parallel
// Applies transform_fn to each element and returns results
pub fn parallel_map[T, U](items []T, config ParallelConfig, transform_fn fn(T) U) []U {
	if items.len == 0 {
		return []U{}
	}

	num_workers := if config.num_workers > 0 { config.num_workers } else { 4 }

	// For small arrays, use sequential processing
	if items.len < config.chunk_size * num_workers || num_workers == 1 {
		mut results := []U{len: items.len}
		for i in 0 .. items.len {
			results[i] = transform_fn(items[i])
		}
		return results
	}

	// Allocate result array
	mut results := []U{len: items.len}

	// Launch workers
	num_workers_actual := if num_workers > items.len { items.len } else { num_workers }
	items_per_worker := items.len / num_workers_actual
	remainder := items.len % num_workers_actual

	ch := chan int{cap: num_workers_actual}

	mut worker_start := 0
	for w in 0 .. num_workers_actual {
		worker_items := items_per_worker + if w < remainder { 1 } else { 0 }
		worker_end := worker_start + worker_items

		go fn (start int, end int, it []T, mut res []U, tf fn(T) U, done chan int) {
			for i in start .. end {
				res[i] = tf(it[i])
			}
			done <- 1
		}(worker_start, worker_end, items, mut &results, transform_fn, ch)

		worker_start = worker_end
	}

	// Wait for all workers
	for _ in 0 .. num_workers_actual {
		_ = <-ch
	}

	return results
}

// get_num_workers - Get optimal number of workers
// Returns a reasonable default (4)
pub fn get_num_workers() int {
	return 4
}

// is_parallelization_beneficial - Check if parallelization is worth the overhead
pub fn is_parallelization_beneficial(num_items int, min_items_per_worker int) bool {
	num_cpus := 4  // Default to 4 workers
	return num_items >= min_items_per_worker * num_cpus && num_cpus > 1
}
