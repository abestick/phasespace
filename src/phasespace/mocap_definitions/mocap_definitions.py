#!/usr/bin/env python
"""
This module provides various classes which simply define the naming conventions and marker quantities for a
particular phasespace setup
"""


class MocapWrist(object):
    NUM_HAND = 4
    NUM_WRIST = 1
    NUM_ARM = 3

    names = ["hand_%d"%h for h in range(NUM_HAND)] + \
            ["wrist_%d"%w for w in range(NUM_WRIST)] + \
            ["arm_%d"%a for a in range(NUM_ARM)]

    groups = {'hand': names[0:NUM_HAND], 'wrist': names[NUM_HAND:NUM_HAND+NUM_ARM], 'arm': names[NUM_HAND+NUM_WRIST:]}

    configs = ['roll', 'pitch', 'yaw']

    def get_marker_group(self, group):
        return self.groups[group]
