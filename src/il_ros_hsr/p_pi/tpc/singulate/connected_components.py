import numpy as np
import math
import cv2
from scipy.ndimage.measurements import label
from union import UnionFind
from skimage.measure import block_reduce
from groups import Group
import IPython

"""returns a list of groups based just on adjacency (no tolerance)"""
def generate_groups(img):
    #give each component a different integer label in the output matrix
    labeled_img = np.zeros(img.shape)

    #all ones --> 8 connectivity (use cross shape for 4 connectivity)
    struct = np.ones((3, 3))
    n_features = label(img, structure = struct, output = labeled_img)

    groups_by_label = {}
    groups = []

    #use labels to put pixels into groups
    for y in range(len(img)):
        for x in range(len(img[0])):
            #foreground pixels
            if img[y][x]:
                curr_label = labeled_img[y][x]

                #create the group if it does not yet exist
                if curr_label not in groups_by_label:
                    groups_by_label[curr_label] = Group(curr_label)
                    groups.append(groups_by_label[curr_label])

                groups_by_label[curr_label].add((y, x))

    return groups

"""
merges until all groups have at least 'tol' distance between them
"""
def merge_groups(groups, tol):
    #give groups initial labels = indexes
    for i in range(len(groups)):
        groups[i].updateLabel(i)

    uf = UnionFind(len(groups))

    #find labels to merge
    for i in range(len(groups)):
        curr_group = groups[i]
        #only look at groups farther in list
        for j in range(i, len(groups)):
            other_group = groups[j]

            #short circuit if already connected (minimize nearest neighbor calls)
            if not uf.find(curr_group.label, other_group.label):
                if Group.nearby(curr_group, other_group, tol):
                    uf.union(curr_group.label, other_group.label)

    merged_groups = []
    unmerged_groups = []
    #iterate until all merges have been made
    while len(groups) > 0:
        curr_group = groups[0]
        merged_groups.append(curr_group)
        for j in range(1, len(groups)):
            other_group = groups[j]

            #on each iteration, one merged group moves to the new array
            if uf.find(curr_group.label, other_group.label):
                curr_group.merge(other_group)
            #all other groups are kept in the old array
            else:
                unmerged_groups.append(other_group)

        groups = unmerged_groups
        unmerged_groups = []

    return merged_groups

"""return a list of the smallest n groups"""
def get_smallest(groups, n):
    groups.sort()
    return groups[:min(n, len(groups))]

"""
input `img` is binarized between fg and bg
two groups are singulated if they are more than `tol` apart
returns centroid and orientation of each group
"""
def singulate(img, tol):
    #halve image size to increase speed
    scale_factor = 2
    area_cutoff = 80
    img = block_reduce(img, block_size = (scale_factor, scale_factor), func = np.mean)
    tol = tol/scale_factor

    #find groups of adjacent foreground pixels
    groups = generate_groups(img)
    groups = [g for g in groups if g.area >= area_cutoff/scale_factor]
    groups = merge_groups(groups, tol)

    center_masses = [map(lambda x: x * scale_factor, g.center_mass) for g in groups]
    directions = [g.orientation() for g in groups]
    # coords = [map(lambda x: x * scale_factor, g.get_bounds()) for g in groups]
    # return coords
    return center_masses, directions
