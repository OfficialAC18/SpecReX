#!/usr/bin/env python3
import cv2
import os
os.environ["XDG_SESSION_TYPE"] = "eglfs"

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks, peak_prominences
import random
import colorsys

#Class for creating infinite number of colors
class ColourCycle:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        h = self.rng.random()
        s = self.rng.uniform(0.5, 1.0)
        v = self.rng.uniform(0.5, 1.0)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))



def spectra_ranking_plot(destination, spectra, wn, ranking, explanations):
    #Make sure that all the arrays are 1-D
    spectra = np.squeeze(spectra)
    wn = np.squeeze(wn)

    assert ranking.shape == wn.shape
    assert spectra.shape == wn.shape

    fig, axs = plt.subplots(nrows=2,
                       ncols=1,
                       figsize=(15,12))

    #Set the facecolor 
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(0.45)

    #Plot the spectra with the wavenumber in the first plot
    #The ranking along with the wavenumber in the second plot
    axs[0].plot(wn,spectra,color='black')
    axs[0].title.set_text('Spectra')
    axs[0].grid(which = 'major', linestyle='-')
    axs[0].set_xlabel('Wavenumber')
    axs[0].set_ylabel('Intensity (A.U)')
    # axs[0].grid(which = 'minor', linestyle='--',alpha = 0.75)
    axs[1].plot(wn,ranking,color='black')
    axs[1].title.set_text('Ranking')
    axs[1].grid(which = 'major', linestyle='-')
    axs[1].set_xlabel('Wavenumber')
    axs[1].set_ylabel('Responsibility')

    #Calculate the alphas for the peaks
    magnitude = ranking/np.max(ranking)
    
    #Get the y-min and set it to be static (For plotting purposes)
    plot_ymin = axs[0].get_ylim()[0]
    axs[0].set_ylim(bottom = plot_ymin)

    #Get the ColourCycle for plotting
    colors = ColourCycle(seed = 42)

    for cause, widths in explanations.items():
        #Get the next color for the cause
        color = next(colors)
        for location, width in zip(cause,widths):
            #0.6 is an arbitrary value, it seems to be the best compromise between visibility of both spectra and peak
            alpha = 0.6*magnitude[location]

            #Generate values for y at the specfic location, with the max being the spectra value at the point
            plot_y_vals = np.linspace(plot_ymin,spectra[location])
            plot_x_vals = np.ones_like(plot_y_vals)*wn[location]
            axs[0].plot(plot_x_vals,
                        plot_y_vals,
                        color = color,
                        alpha = alpha,
                        linewidth = 1)
            
            for i in range(-width,width):
                if location + i < len(spectra.reshape(-1)):
                    plot_y_vals = np.linspace(plot_ymin,spectra[location+i])
                    plot_x_vals = np.ones_like(plot_y_vals)*(wn[location+i])
                axs[0].plot(plot_x_vals,
                            plot_y_vals,
                            color = color,
                            alpha = alpha,
                            linewidth = 1)

    #Similarily for the ranking plot
    plot_ymin = axs[1].get_ylim()[0]
    axs[1].set_ylim(bottom = plot_ymin)

    #Reset ColourCycle
    colors = ColourCycle(seed = 42)

    for cause, widths in explanations.items():
        #Get the next color for the cause
        color = next(colors)
        labels = []
        label = ''
        for idx, (location, width) in enumerate(zip(cause,widths)):
            alpha = 0.6*magnitude[location]
            #Generate values for y at the specfic location, with the max being the spectra value at the point
            plot_y_vals = np.linspace(plot_ymin,ranking[location])
            plot_x_vals = np.ones_like(plot_y_vals)*wn[location]
            #Create the label for the example
            labels.append((location, width))

            if idx == len(cause) - 1:
                for i in range(len(labels)):
                    if i > 0:
                        label += f" $\wedge$ (${int(wn[labels[i][0]] - labels[i][1])} - {int(wn[labels[i][0]] + labels[i][1])}$)"
                    else:
                        label += f"(${int(wn[labels[i][0]] - labels[i][1])} - {int(wn[labels[i][0]] + labels[i][1])}$)"

                axs[1].plot(plot_x_vals,
                            plot_y_vals,
                            color = color,
                            alpha = alpha,
                            linewidth = 1,
                            label = label)
                
            for i in range(-width,width):
                if location + i < len(spectra.reshape(-1)):
                    plot_y_vals = np.linspace(plot_ymin,ranking[location+i])
                    plot_x_vals = np.ones_like(plot_y_vals)*(wn[location+i])
                    
                axs[1].plot(plot_x_vals,
                            plot_y_vals,
                            color = color,
                            alpha = alpha,
                            linewidth = 1)

    #Now add the legend
    #First get the handles and labels
    handles, labels = axs[1].get_legend_handles_labels()
    pos = axs[0].get_position()
    fig.legend(handles=handles, labels = labels, loc = [pos.x0 + 0.68, pos.y0 - 0.05], title = "Model Explanations")
    fig.tight_layout()
    fig.subplots_adjust(right = 0.70)   

    if destination is None:
        fig.show()
    else:
        #Save the plot
        fig.savefig(
            destination,
            dpi = 900,
            bbox_inches = "tight"
        )

