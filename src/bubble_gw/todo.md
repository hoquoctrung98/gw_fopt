+ Inside compute_t_rr, don't add k_squared as an argument
+ In the beginning of compute_k_integral, don't compare float number.
+ Expose t_zz, t_rr, t_xz to python
+ Add different types of cut-off
+ Add cut-off in computation of bulk-flow
+ Add wall velocity to the computation
+ Add simulation for field evolution to two_bubbles.
+ PyArray::as_array to avoid to_owned_array

Use a Contiguous Point Buffer:Problem: grid[[i, k]] accesses are non-contiguous due to ndarray’s row-major layout and non-sequential i values. The Array2<f64> stores points as [x0, y0, z0, x1, y1, z1, ...], but accessing grid[[i, 0]], grid[[i, 1]], grid[[i, 2]] separately may split cache lines.
Solution: Replace grid: Array2<f64> with a contiguous Vec<f64> where each point’s [x, y, z] is stored consecutively (stride 3). Access points as grid[i*3], grid[i*3+1], grid[i*3+2].
How: Modify Lattice::generate_grid to return a Vec<f64> instead of Array2<f64>. Update BubbleFormationSimulator to store grid: Vec<f64> and adjust indexing in get_valid_points, update_outside_mask, and other methods (e.g., get_center, generate_exterior_bubbles).
Impact: Contiguous access to [x, y, z] (24 bytes) fits within a single cache line, reducing cache misses compared to ndarray’s potential column-wise jumps.
Trade-off: Requires refactoring all grid accesses, but the change is straightforward and preserves logic. Memory usage remains the same (~8 MB).
Cache-Aligned Data Structures:Problem: The Bubble struct (center: [f64; 3], time: f64) may not be cache-aligned, leading to partial cache line usage or false sharing in parallel loops.
Solution: Ensure Bubble is aligned to cache line boundaries (typically 64 bytes) using padding or custom allocation. Alternatively, split bubbles into separate Vec<f64> for centers and times to improve access patterns.
How: Use #[repr(align(64))] on Bubble or store bubbles as Vec<f64> (e.g., [x0, y0, z0, t0, x1, y1, z1, t1, ...]). Update access patterns in get_valid_points and other methods.
Impact: Ensures each Bubble access fully utilizes a cache line, reducing misses and false sharing in parallel loops.
Trade-off: Increases memory usage slightly due to padding or restructuring but simplifies access patterns.

