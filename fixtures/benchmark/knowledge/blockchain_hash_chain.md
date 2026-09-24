# Blockchain Hash-Chain Structure

A blockchain is a decentralized, tamper-evident ledger composed of sequentially linked blocks of transactional data.
Each block contains a cryptographic hash of its transactions (Merkle tree root), a timestamp, a nonce, and the cryptographic hash of the immediate previous block.
Because each block cryptographically references the prior block's hash, altering any historical transaction invalidates all subsequent block hashes.
Consensus mechanisms like Proof of Work enforce chronological consistency and immutability across distributed network nodes.
