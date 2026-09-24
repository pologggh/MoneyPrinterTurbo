# Transformer Attention Mechanism

The Transformer architecture relies on the Multi-Head Self-Attention mechanism to compute representations of input sequences without using recurrent layers.
In self-attention, each input token is projected into Query (Q), Key (K), and Value (V) vector spaces.
Attention scores are computed as the scaled dot-product between Queries and Keys, divided by the square root of key dimension d_k, followed by softmax normalization.
This enables parallelization across long-range sequence dependencies and captures contextual relationships across tokens.
