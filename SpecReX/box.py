#!/usr/bin/env python
'''
Modified Code from Private Code Repository:
Multiple Different Black Box Explanations for Image Classifiers by Chockler et al.
https://arxiv.org/abs/2309.14309

'''


from __future__ import annotations

from typing import List
from anytree import LevelOrderGroupIter, NodeMixin, RenderTree, PreOrderIter
import numpy as np

from SpecReX.distributions import random_pos

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
        return (self.row_stop - self.row_start,)

    def spawn_children(self, min_size) -> List[Box]:
        """split a box into 4 contiguous sections"""
        if self.length() < min_size:
            return []
    
        row_mid = random_pos(self.distribution, [self.row_start, self.row_stop, self.distribution_args])
        row_lt = random_pos(self.distribution, [self.row_start, row_mid-1, self.distribution_args])
        row_gt = random_pos(self.distribution, [row_mid+1, self.row_stop, self.distribution_args])

        children = []
        counter = 0

        if row_lt is not None:
            b0 = Box(
                self.row_start,
                row_lt,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
                mutant = None 
            )
            b0.update_name(f":{counter}")
            counter += 1

            children.append(b0)
            

            if row_mid is not None:
                b1 = Box(
                    row_lt,
                    row_mid,
                    distribution=self.distribution,
                    distribution_args=self.distribution_args,
                    name=self.name,
                    mutant = None 
                )
                b1.update_name(f":{counter}")
                counter += 1
            
            children.append(b1)
        
        if row_mid is not None and row_gt is not None:
            b2 = Box(
                row_mid,
                row_gt,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
                mutant = None 
            )
            b2.update_name(f":{counter}")
            counter += 1

            b3 = Box(
                row_gt,
                self.row_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
                mutant = None 
            )
            b3.update_name(f":{counter}")
            counter += 1

            children.append(b2)
            children.append(b3)

        return children

    def length(self):
        return self.row_stop - self.row_start

    def apply_to_mask(self, current_mask):
        """set everything in the bounding box to True"""
        if current_mask.shape[0] == 3 or current_mask.shape[0] == 1:
            current_mask[:, self.row_start : self.row_stop] = True
        else:
            current_mask[self.row_start : self.row_stop] = True

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
        mutant = None,
    ) -> None:
        super().__init__(row_start, row_stop, distribution, distribution_args, name)
        self.parent = parent
        self.children = children
        self.mutant = mutant

    def add_children_to_tree(self, min_size):
        if not self.children:
            self.children = self.spawn_children(min_size)


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


def build_tree(root, depth, min_size) -> None:
    for n in PreOrderIter(root):
        if n.depth <= depth and len(n.children) == 0:
            n.add_children_to_tree(min_size)
