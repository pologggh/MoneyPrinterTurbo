# Relational Database Indexing

Database indexes are specialized data structures that improve the speed of data retrieval operations on database tables at the cost of additional storage and write latency.
The most common relational index structure is the balanced B-Tree (or B+ Tree), which maintains sorted key-pointer pairs with logarithmic O(log N) lookup and range scan complexity.
Without indexes, database engines must perform sequential full-table scans O(N) to locate matching rows.
