import os
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from tqdm import tqdm

#We Define the Double Peak Spectra as Following
    #The Postive Class has 2 peaks.
    #The 2 largest scaled products of height and width are the discriminative peaks.
    #The position of the positive peaks is > 100 from each other, but < 300.
    #The noise peaks can be anywhere.
    #All peaks are atleast 60 

DATASET_DIR = "/home/akchunya/Akchunya/MSc Thesis/Simulated Spectra Dataset/Double Peak"

def generate_spectrum(
        num_peaks = 4,
        num_important_peaks = 2,
        num_examples_gen = 10,
        min_peak_width = 5,
        max_peak_width = 20,
        max_peak_height = 4000,
        min_peak_height = 2000,
        num_baseline_pts = 8,
        noise_factor = 20,
        baseline_cap = 2000,
        peak_cap = 30,
        spectra_length = 900,
):
    
    """Generates simulated spectrum with the provided parameters, creates a number of examples given the same parameters.
    The spectra is SNV normalized before it is returned.
    
    Parameters
    ----------
    num_peaks : int, optional
        The maximum number of peaks that will be present in the spectra (default is 4).
    num_important_peaks : int, optional
        The number of peaks that are important for classfication.
    num_examples_gen : int, optional
        The number of examples that will be generated based on the provided parameters (default is 10).
    min_peak_width: int, optional
        The minimum width of a given peak in the spectrum (default is 5).
    max_peak_width: int, optional
        The maximum width of a given peak in the spectrum (default is 20).
    max_peak_height: int, optional
        The maximum height a peak can reach (will be scaled according to a scaling factor) (default is 4000).
    min_peak_height: int, optional
        The minimum height a peak can reach (will be scaled according to a scaling factor) (default is 2000).
    num_baseline_pts: int, optional
        The number of pts used to fit the CubicSpline, needs to be greater than 2 (default is 8).
    baseline_cap: int, optional
        The maximum value for baseline pts (default is 2000).
    peak_cap: int, optional
        The limiting distance for peaks. Peak postions will be capped to spectra_length - peak_cap (default is 30).
    spectra_length: int, optional
        The length of the simulated spectra.
    
    Returns
    -------
    snv_norm_spectra_pos
        a numpy array of positive SNV normalized spectra.
    snv_norm_spectra_neg
        a numpy arrat of negative SNV normalized spectra.
    """

    assert num_baseline_pts >=2, "Number of baseline points must be greater than or equal to 2 for performing interpolation"
    assert num_important_peaks < num_peaks, "Number of important peaks must be lower than the total number of peaks in spectra"

    #Generate the peak widths
    widths = np.random.randint(min_peak_width, max_peak_width + 1, num_peaks)

    #Generate the peak heights
    heights = np.random.randint(min_peak_height, max_peak_height + 1, num_peaks)

    #Calculate Peak Importance (used for creating positives and negatives)
    peak_importance = np.argsort(widths * heights)[::-1]
    
    #Classify which points are the positives 
    positve_peaks = peak_importance[:num_important_peaks]
    negative_peaks = peak_importance[num_important_peaks:]

    # print("Positive Peaks:",positve_peaks)
    # print("Negative Peaks:",negative_peaks)

    #Lists to store the examples
    pos_examples = []
    neg_examples = []

    #Now we can generate the required spectras
    #This is to make sure that the number of positive and negative examples are roughly equal
    while len(pos_examples) != num_examples_gen or len(neg_examples) != num_examples_gen:
        #This is in order to make our calculations easier
        num_positive = len(positve_peaks)
        num_negative = len(negative_peaks)

        #Determine if this is a positive or negative example
        pos_example = 0 if np.random.randint(0,2) == 0 else 1

        #Now, we need to generate the positions for the spectra
        #If it is positive, we follow the positive spectra criterion (min position difference - 100, max position difference - 300)
        #If it is negative, the discriminating peaks are out of the postive spectra region
        if pos_example:
            while True:
                min_dist = spectra_length
                max_dist = -1
                #Generate the random positions for positive positions
                positive_positions = np.random.randint(50, spectra_length - peak_cap, num_positive)

                #Calculate the distance amongst the spectra
                for i in range(len(positive_positions)):
                    for j in range(i+1,len(positive_positions)):
                        dist = abs(positive_positions[i] - positive_positions[j])
                        if dist < min_dist:
                            min_dist = dist
                        if dist > max_dist:
                            max_dist = dist

                if min_dist > 100 and max_dist < 300:
                    break

            # print("Min Distance (Pos):",min_dist)
            # print("Max Distance (Pos):",max_dist)
            
        else:
            while True:
                min_dist = spectra_length
                max_dist = -1
                #Generate the random positions for positive positions
                positive_positions = np.random.randint(50, spectra_length - peak_cap, num_positive)

                #Calculate the distance amongst the spectra
                for i in range(len(positive_positions)):
                    for j in range(i+1,len(positive_positions)):
                        dist = abs(positive_positions[i] - positive_positions[j])
                        if dist < min_dist:
                            min_dist = dist
                        if dist > max_dist:
                            max_dist = dist

                if  60 <= min_dist < 100 and max_dist > 300 or \
                    60 <= min_dist < 100 and 60 <= max_dist < 100 or \
                    min_dist > 300 and max_dist > 300:
                       break              
            
            # print("Min Distance (Neg):",min_dist)
            # print("Max Distance (Neg):",max_dist)

        #Generate the negative peak postions, make sure they don't overlap with positive postions
        #Also maintain some distance between two peaks (Using 30)
        while True:
            min_dist = spectra_length
            negative_postions = np.random.randint(50,spectra_length - peak_cap, num_negative)
            all_pos = np.concatenate([positive_positions,negative_postions],axis = 0)
            for i in range(len(all_pos)):
                for j in range(i+1,len(all_pos)):
                    dist = abs(all_pos[i] - all_pos[j])
                    if dist < min_dist:
                        min_dist = dist

            if min_dist >= 60 and np.intersect1d(negative_postions,positive_positions).size == 0:
                break

        
        #Now we generate the required spectra
        #Values for generating the spectra
        x_vals = np.linspace(0, spectra_length, spectra_length)
        peak_vals = np.zeros_like(x_vals)

        # print("All Pos",all_pos)

        for position, width, height in zip(all_pos,widths,heights):
            #We calculate the scaling factors for the height, based on the PDF value of the Normal distribution
            #at the particular width and height
            scaling_factor = height * np.sqrt(2 * np.pi) * width

            #Apply the scaling factor to the Gaussian Function
            peak_vals += scaling_factor * norm.pdf(x_vals,position,width)

        #Generate Noise
        noise = noise_factor * np.random.normal(size = x_vals.shape)

        #Calculate baseline positions
        x_base = np.linspace(0,baseline_cap,num_baseline_pts)
        
        #We also calculate an additional bias term to scale the y vals (Might not be needed)
        x_bias = np.linspace(baseline_cap,0,num_baseline_pts)

        #Calculate the y-values for the baseline
        y_base = np.random.rand(num_baseline_pts) * x_bias

        #Fit the curve and get the values for the same
        spectra = CubicSpline(x_base,y_base)(x_vals)

        #Now add the peaks and noise
        spectra += peak_vals + noise

        #SNV Normalize the spectra
        mean = np.mean(spectra)
        std = np.std(spectra)
        spectra = (spectra - mean)/std

        #Append to appropriate list
        if pos_example and len(pos_examples) < num_examples_gen:
            pos_examples.append(spectra)
        elif not pos_example and len(neg_examples) < num_examples_gen:
            neg_examples.append(spectra)

        # print("Number of Positive Examples:",len(pos_examples))
        # print("Number of Negative Examples:",len(neg_examples))

    return np.stack(pos_examples,axis = 0), np.stack(neg_examples,axis = 0)


#Generate Test Spectra
params = {
        "num_peaks": 4,
        "num_important_peaks": 2,
        "num_examples_gen":100,
        "min_peak_width": 5,
        "max_peak_width":20,
        "max_peak_height": 4000,
        "min_peak_height": 2000,
        "num_baseline_pts": 8,
        "noise_factor":20,
        "baseline_cap":2000,
        "spectra_length":900,
}


#We shall restart the call to generate spectrum every iteration in order to get fresh widths and heights
#Number of examples = Iterations * num_examples_gen
pos_counter = 1
neg_counter = 1
for _ in tqdm(range(1000)):
    pos_spectra, neg_spectra = generate_spectrum(**params)

    for i in range(len(pos_spectra)):
        np.save(os.path.join(DATASET_DIR,f"Positive/positive{pos_counter}.npy"),pos_spectra[i])
        pos_counter += 1

    for i in range(len(neg_spectra)):
        np.save(os.path.join(DATASET_DIR,f"Negative/negative{neg_counter}.npy"),neg_spectra[i])
        neg_counter += 1 





    
