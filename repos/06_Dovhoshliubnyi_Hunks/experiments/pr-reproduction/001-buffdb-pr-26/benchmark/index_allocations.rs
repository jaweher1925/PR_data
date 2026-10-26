use buffdb::index::{IndexConfig, IndexManager, IndexType};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const ITERATIONS: usize = 100_000;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn run(index_count: usize) {
    let manager = IndexManager::new();
    for index in 0..index_count {
        manager
            .create_index(IndexConfig {
                name: format!("index_{index}"),
                index_type: if index % 2 == 0 {
                    IndexType::Hash
                } else {
                    IndexType::BTree
                },
                unique: false,
                filter: None,
            })
            .unwrap();
    }

    let key = "primary-key-".repeat(8);
    let value_a = "indexed-value-a-".repeat(32);
    let value_b = "indexed-value-b-".repeat(32);
    manager.update_indexes(&key, None, &value_a).unwrap();

    for iteration in 0..10_000 {
        let (old, new) = if iteration % 2 == 0 {
            (&value_a, &value_b)
        } else {
            (&value_b, &value_a)
        };
        manager.update_indexes(&key, Some(old), new).unwrap();
    }

    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let start = Instant::now();
    for iteration in 0..ITERATIONS {
        let (old, new) = if iteration % 2 == 0 {
            (&value_a, &value_b)
        } else {
            (&value_b, &value_a)
        };
        black_box(manager.update_indexes(black_box(&key), Some(black_box(old)), black_box(new)))
            .unwrap();
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
    println!(
        "indexes={index_count},iterations={ITERATIONS},elapsed_ns={},ns_per_update={:.2},allocations={allocations},allocations_per_update={:.2}",
        elapsed.as_nanos(),
        elapsed.as_nanos() as f64 / ITERATIONS as f64,
        allocations as f64 / ITERATIONS as f64
    );
}

fn main() {
    for index_count in [1, 2, 10] {
        run(index_count);
    }
}

