# TCP Three-Way Handshake

The Transmission Control Protocol (TCP) establishes a reliable, bidirectional transport connection through a three-way handshake.
First, the client sends a SYN packet with an initial sequence number (ISN) to initiate synchronization.
Second, the server replies with a SYN-ACK packet, acknowledging the client's sequence number and providing its own ISN.
Third, the client sends an ACK packet confirming receipt, transitioning both endpoints into the ESTABLISHED state for reliable data transfer.
