# Domain Name System (DNS) Resolution

DNS resolution translates human-readable domain names into machine-routable IP addresses.
The resolution process begins at a local recursive resolver, which queries the Root DNS servers if the address is not cached.
The Root server delegates the query to the Top-Level Domain (TLD) server (such as .com or .org).
Finally, the TLD server directs the resolver to the authoritative nameserver holding the precise A or AAAA record for the domain.
