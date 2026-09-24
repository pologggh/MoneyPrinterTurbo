# CPU Cache Hierarchy

Modern microprocessors use a multi-tiered hierarchical cache system (L1, L2, and L3 caches) to bridge the speed disparity between fast processor cores and slower main memory (RAM).
L1 cache is split into instruction and data caches, operating at core clock speeds with single-cycle access latency but limited storage capacity.
L2 and L3 caches provide progressively larger capacities with slightly higher latency.
Caching effectiveness relies on spatial and temporal locality principles, predicting memory accesses to prevent memory-stall pipeline bubbles.
