Peers parameters are used to instanciate Peers object with different values.

The different conf files contains a different Peer configuration, good is the base behavior with a good success rate, bad is the opposite and so on.

Refer to simulation default for the following parameters explanation :

- transaction_duration
- capacity
- peer_interactions
- breaking_rate

The **behavior** parameter is a list elements with the format <timestamp>: <success_rate> used to determine the success_rate of a peer.

The following behavior mean that the peer will have 95% success rate starting from the begining and until the end of the simulation.
behavior:
0.0: 0.95

While this one start the same, it success rate drop to 0 at timestamp 6.0 and is back to it's initial value at timestamp 14.0 and until the end of the simulation.
behavior:
0.0: 0.95
6.0: 0.0
14.0: 0.95
