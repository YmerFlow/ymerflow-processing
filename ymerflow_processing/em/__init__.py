"""Shared by both electromagnetic methods — time-domain and frequency-domain.

What belongs here is anything that assumes a transmitter and a receiver but not
a particular measurement domain: transmitter attitude correction, the horizontal
transmitter assumption, forward-model normalization, filling current from the
system description, and applying the system's own gate mask.

These are placed here rather than in ``tem/`` specifically so that frequency-domain
work can use them unchanged, rather than importing across from a sibling or
duplicating them.
"""
