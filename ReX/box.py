#!/usr/bin/env python
from __future__ import annotations

from typing import List
from anytree import LevelOrderGroupIter, NodeMixin, RenderTree, PreOrderIter
import numpy as np

from ReX.distributions import Distribution, random_pos

from ReX.specaug import split_and_interpolate


class BoxInternal:
    def __init__(
        self,
        row_start,
        row_stop,
        distribution=None,
        distribution_args=None,
        name="",
    ) -> None:
        self.name = name
        self.distribution = distribution
        self.distribution_args = distribution_args
        self.row_start = row_start
        self.row_stop = row_stop

    def __repr__(self) -> str:
        return f"Box < name: {self.name}, row_start: {self.row_start}, row_stop: {self.row_stop}, length: {self.length()}>"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.name == other
        elif not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def update_name(self, name: str):
        self.name += name

    def shape(self):
        return (self.row_stop - self.row_start,) #self.col_stop - self.col_start)

    def spawn_children(self, min_size, invert, pos_ranking=None) -> List[Box]:
        """split a box into 4 contiguous sections"""
        if self.length() < min_size:
            return []
        
        if self.distribution == Distribution.Adaptive and pos_ranking is not None:
            l = pos_ranking.shape
            mask = np.zeros(l)
            mask[self.row_start : self.row_stop] = pos_ranking[self.row_start : self.row_stop]
            row_mid = random_pos(self.distribution, [self.row_start, self.row_stop, self.distribution_args])
            row_lt = random_pos(self.distribution, [self.row_start, row_mid-1, self.distribution_args])
            row_gt = random_pos(self.distribution, [row_mid+1, self.row_stop, self.distribution_args])  # type: ignore
        else:
            #Would need a better strategy here
            row_mid = random_pos(self.distribution, [self.row_start, self.row_stop, self.distribution_args])
            row_lt = random_pos(self.distribution, [self.row_start, row_mid-1, self.distribution_args])
            row_gt = random_pos(self.distribution, [row_mid+1, self.row_stop, self.distribution_args])

            if row_mid is None or row_lt is None or row_gt is None:
                return []

        b0 = Box(
            self.row_start,
            row_lt,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b0.update_name(":0")

        b1 = Box(
            row_lt,
            row_mid,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b1.update_name(":1")

        b2 = Box(
            row_mid,
            row_gt,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b2.update_name(":2")

        b3 = Box(
            row_gt,
            self.row_stop,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b3.update_name(":3")

        return [b0, b1, b2, b3]

    #For SpecRex
    def length(self):
        return self.row_stop - self.row_start

    def remove_from_mask(self, current_mask):
        """set everything in the bounding box to False"""
        current_mask[self.row_start : self.row_stop] = False

    # def apply_to_mask(self, current_mask): #Instead of this, we perform split and interpolate
    #     """set everything in the bounding box to True"""
    #     if current_mask.shape[0] == 3:
    #         current_mask[:, self.row_start : self.row_stop] = True
    #     else:
    #         current_mask[self.row_start : self.row_stop] = True

    def interpolate_mask(self, 
                         current_mask,
                         wavenumber,
                         spec_shape,
                         method='linear'):
        '''
        Interpolate the mask at the bounds of the child node

        args:
            current_mask - The current mask, with all previous interpolations applied
            wavenumber - The wavenumbers associated with the spectra
            method - Type of interpolation to be performed (linear, cubic (splines))
        '''
        current_mask = split_and_interpolate(wavenumber=wavenumber,
                                             spectra=current_mask,
                                             spec_shape=spec_shape,
                                             r_start = self.row_start,
                                             r_lim=self.row_stop,
                                             method=method)


#We use row_start/row_stop | col_start/col_stop as the regions where we interpolate
class Box(BoxInternal, NodeMixin):
    def __init__(
        self,
        row_start,
        row_stop,
        distribution=None,
        distribution_args=[],
        name="",
        parent=None,
        children=[],
    ) -> None:
        super().__init__(row_start, row_stop, distribution, distribution_args, name)
        self.parent = parent
        self.children = children

    def add_children_to_tree(self, min_size, invert, pos_ranking=None):
        if not self.children:
            self.children = self.spawn_children(min_size, pos_ranking=pos_ranking, invert=invert)


def initialise_tree(r_lim, distribution, distribution_args, r_start=0) -> Box:
    return Box(r_start, r_lim, distribution, distribution_args, name="r")


def show_tree(tree):
    print(RenderTree(tree))


def average_box_length(tree, d) -> float:
    lengths = [[node.length() for node in children] for children in LevelOrderGroupIter(tree)]
    try:
        return np.mean(lengths[d], axis=0)
    except IndexError:
        return 0.0


def box_dimensions(box: Box):
    return (box.row_start, box.row_stop)


def boxes_name_and_dimensions(boxes: List[Box]):
    return [(box.name, box_dimensions(box)) for box in boxes]


def build_tree(root, depth, min_size, pos_ranking=None, invert=True) -> None:
    for n in PreOrderIter(root):
        if n.depth <= depth and len(n.children) == 0:
            n.add_children_to_tree(min_size, invert, pos_ranking=pos_ranking)
