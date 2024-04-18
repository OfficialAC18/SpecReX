#!/usr/bin/env python3
import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import center_of_mass

# from ReX.logger import logger


def resize_image(pic, size):
    img = cv2.imread(pic)
    if len(size) == 3:
        img = cv2.resize(img, dsize=(size[1], size[2]), interpolation=cv2.INTER_CUBIC)
        img = img.transpose(2, 0, 1)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def heatmap(original, destination, pixel_ranking):
    """overlays a heatmap on image <original>, saving to <destination> using <pixel_ranking> information"""
    img = resize_image(original, pixel_ranking.shape)
    heatmap = cv2.normalize(pixel_ranking, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    cv2.imwrite(destination, superimposed)


def plot_3d(path, ranking, ogrid):
    img: NDArray = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ranking.shape[0], ranking.shape[1]), interpolation=cv2.INTER_AREA)
    img = img / 255  # type: ignore
    if ogrid:
        X, Y = np.ogrid[0 : img.shape[0], 0 : img.shape[1]]
    else:
        X, Y = np.meshgrid(np.arange(0, ranking.shape[0], 1), np.arange(0, ranking.shape[1], 1))
    return img, X, Y


def contour_plot(path, ranking, levels=10, destination=None):
    img, X, Y = plot_3d(path, ranking, False)
    _, ax = plt.subplots()
    ax.imshow(img, zorder=1, interpolation="bilinear")
    ax.contourf(X, Y, ranking, levels=levels, zorder=2, alpha=0.6, cmap=cm.coolwarm)  # type: ignore
    plt.axis("off")
    if destination is None:
        plt.show()
    else:
        plt.savefig(destination)


def surface_plot(path, ranking, destination=None, decorate=True):
    img, X, Y = plot_3d(path, ranking, True)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, np.atleast_2d(0), rstride=5, cstride=5, facecolors=img)  # type: ignore
    ax.plot_surface(X, Y, ranking, alpha=0.4, cmap=cm.coolwarm)  # type: ignore
    if decorate:
        x, y = center_of_mass(ranking)
        x = int(round(x))
        y = int(round(y))
        z = ranking[x, y]
        ax.scatter(x, y, z, color="b")
        ax.text(x, y, z, s="center of mass")  # type: ignore
        loc = np.unravel_index(np.argmax(ranking), ranking.shape)
        ax.scatter(loc[0], loc[1], ranking[loc[0], loc[1]], color="r")
        ax.text(loc[0], loc[1], ranking[loc[0], loc[1]], s="max point")  # type: ignore

    if destination is None:
        plt.show()
    else:
        plt.savefig(destination)


def produce_image(args, pixel_ranking):
    if args.surface is not None:
        if args.surface == "show":
            surface_plot(args.path, pixel_ranking)
        else:
            surface_plot(args.path, pixel_ranking, destination=args.surface)
    if args.contour is not None:
        if args.contour == "show":
            contour_plot(args.path, pixel_ranking)
        else:
            contour_plot(args.path, pixel_ranking, destination=args.surface)
    if args.heatmap is not None:
        if args.heatmap == "show":
            heatmap(args.path, "heatmap.jpg", pixel_ranking)
        else:
            heatmap(args.path, args.heatmap, pixel_ranking)


def masked_image(path, destination, explanation, mask_value, processed=True):
    if processed:
        img = cv2.imread(path)
        img = np.where(explanation, img, mask_value)
    else:
        img = resize_image(path, explanation.shape)
        img = np.where(explanation, img, mask_value)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    cv2.imwrite(destination, img)  # type: ignore


### Debugging functions ###


def print_mask(mask, destination):
    blank_image = np.zeros((3, mask.shape[1], mask.shape[2]), np.uint8)
    # blank_image[:] = (225, 225, 225)
    img = np.where(mask, blank_image, 0)
    cv2.imwrite(destination, img)


def print_spatial_mask(mask, destination, r, c, radius):
    blank_image = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    # blank_image[:] = (225, 225, 225)
    img = np.where(mask, blank_image, 0)
    x, y = int(c - radius), int(r - radius)
    xw, yh = int(c + radius), int(r + radius)
    cv2.rectangle(img, (x, y), (xw, yh), color=(100, 100, 100))
    cv2.imwrite(destination, img)


def image_debug(path, destination, explanation):
    img = resize_image(path, (224, 224, 3))
    img = np.where(explanation, img, 0)
    cv2.imwrite(destination, img)
