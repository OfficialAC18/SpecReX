
"""Extract Spectral explanations from the responsibility map"""

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from itertools import combinations

from SpecReX.specaug import interpolate_mask

def generate_combinations(lst):
        """
        Create all possible combinations of the examples in the list
        """
        all_combinations = []
        for c in range(1, len(lst) + 1):
                all_combinations.extend(combinations(lst,c))
        
        return all_combinations


def spec_masking(peaks, beam_size, shape):
        """
        Mask the regions that need to be interpolated
        """
        mask = np.zeros(shape = shape)
        for peak in peaks:
                #In case the peak is near the start of the spectra
                if peak < beam_size:
                        mask[0:beam_size] = 1

                #In case the peak is near the end of the spectra
                elif peak + beam_size > shape[0] - 1:
                        mask[shape[0]-1-beam_size:shape[0]] = 1
                
                #In case the peak is somewhere else in the spectra
                else:
                        mask[peak - beam_size: peak + beam_size] = 1
        
        return mask.reshape(1,1,-1)

def extraction(all_causes,
               beam_size,
               beam_eta,
               shape,
               wn_array,
               spec_array,
               interp_method,
               prediction_func,
               target_class):

        """
        Attain the smallest possible parts of causes in order to get the minimal explanation
        """
        all_cause_beam_width = {}

        for cause in tqdm(all_causes,"Reducing Causes"):
                #Now keep testing till we get the minimal explanations
                #Vary one of the causes till the explanation changes, then
                #Fix it and decrease the others till we get the minimal size
                #for all the beams
                cause_beam_width = []
                for idx in range(len(cause)):
                        if len(cause) > 1:
                                if idx < len(cause)-1:
                                        fixed_mask = spec_masking(peaks = cause[0:idx] + cause[idx+1:], beam_size=beam_size,shape = shape)
                                else:
                                        fixed_mask = spec_masking(peaks = cause[0:idx], beam_size=beam_size, shape = shape)
                        else:
                                fixed_mask = np.zeros(shape = shape).reshape(1,1,-1)

                        num_iter = 1
                        while True:
                                if beam_size - num_iter*beam_eta == 0:
                                        cause_beam_width.append(beam_size - (num_iter - 1)*beam_eta)
                                        break
                                var_mask = spec_masking(peaks = [cause[idx]], beam_size = beam_size - num_iter*beam_eta, shape = shape)
                                total_mask = fixed_mask + var_mask

                                potential_cause = interpolate_mask(
                                        total_mask,
                                        wn_array,
                                        spec_array,
                                        interp_method
                                )

                                #If this fails, that means previous one was the optimal length
                                if prediction_func(potential_cause)[0] != target_class:
                                        cause_beam_width.append(beam_size - (num_iter - 1)*beam_eta)
                                        break
                                else:
                                        num_iter += 1
                
                #Since we now have the optimal width for all the causes, we store them in the dictionary
                #The keys are the indexes of the peaks
                all_cause_beam_width[tuple(cause)] = cause_beam_width
        
        return all_cause_beam_width

                


def fixed_beam_search(spec_array,
        wn_array,
        prediction_func,
        pos_ranking,
        interp_method,
        beam_size,
        beam_engulf_window,
        beam_eta,
        responsibility_similarity,
        maxima_scaling_factor,
        multiple,
        target_class,
):
    
        """
        Perform Fixed-Beam Search over the spectral landscape and extract viable explanantions
        both which are independent and conjunctive
        """

        #Calculate the maxima of the responsbility values
        #Engulf peaks which are within some window
        peaks = find_peaks(x = pos_ranking, distance=beam_engulf_window)[0]
        peak_heights = pos_ranking[peaks]

        #Sort the peaks according to peak heights
        peak_pos_desc = np.argsort(peak_heights)[::-1]
        peaks, peak_heights = peaks[peak_pos_desc], peak_heights[peak_pos_desc]

        #Drop peaks if they are below previous maxima by scaling factor
        idx = 0
        idx_rabbit = 1
        drop_indices = []
        
        #To determine whether to continue finding more explanations (In the case of Single/Multi)
        find_explanations = True

        while idx < len(peaks):
                if idx + idx_rabbit < len(peaks) and maxima_scaling_factor*peak_heights[idx] > peak_heights[idx+idx_rabbit]:
                        drop_indices.append(idx+idx_rabbit)
                        idx_rabbit+=1
                else:
                        idx += idx_rabbit
                        idx_rabbit = 1

        #Drop peaks that are not required
        peaks, peak_heights = np.delete(peaks,drop_indices), np.delete(peak_heights,drop_indices)

        #Test Causes
        idx = 0
        all_causes = []
        while idx < len(peaks) and find_explanations:
                #First check if the peak is a cause
                #For that interpolate out the other regions
                #If it doesn't match, check if there are similar responsibility peaks
                #and see if they are conjunctive parts of the cause
                #For computational tractability, we limit it to atmost 5 causes

                sim_resp = []
                exceeds_threhold = False
                idx_rabbit = 1
                while not exceeds_threhold and len(sim_resp) < 4:
                        if idx + idx_rabbit < len(peaks) and (peak_heights[idx + idx_rabbit]/peak_heights[idx]) >= responsibility_similarity:
                                sim_resp.append(peaks[idx + idx_rabbit])
                                idx_rabbit += 1
                        else:
                                exceeds_threhold = True
                
                #Add the peak itself to the array of similar responsibility values
                sim_resp.append(peaks[idx])

                #Move the idx to the rabbit position
                idx += idx_rabbit

                #Generate combinations of the similar responsbility peaks
                combinations = generate_combinations(sim_resp)

                #Go through the generated combinations and test the possible combinations
                #If a set passes, remove all the possible combinations of which it is a subset (since otherwise the cause is not minimal)
                removed_subsets = []
                for combination in combinations:
                        #Check if the combination is a superset of an existing cause
                        if combination not in removed_subsets:
                                #Create the required mask
                                mask = spec_masking(peaks=combination,
                                                beam_size=beam_size,
                                                shape=pos_ranking.shape)
                                

                                #Create the required mutant
                                potential_cause = interpolate_mask(mask=mask,
                                                                wavenumber=wn_array,
                                                                spectra=spec_array,
                                                                method=interp_method)
                                
                                #Test the mutant
                                if prediction_func(potential_cause)[0] == target_class:
                                        all_causes.append(list(combination))

                                        #Stop after one explanation if is not multiple
                                        if not multiple:
                                                find_explanations = False
                                                break
                                        
                                        #Filter all supersets of this particular combination
                                        removed_subsets.extend(list(filter(lambda x: set(combination).issubset(x),combinations)))
                                

        #Now that we have the causes, it's time to ablate them to extract a more optimized solution
        explanation_coords = extraction(all_causes,
                                        beam_size,
                                        beam_eta,
                                        pos_ranking.shape,
                                        wn_array,
                                        spec_array,
                                        interp_method,
                                        prediction_func,
                                        target_class)
        
        #Return the coords, visualization handles the rest
        return explanation_coords




                                

                                                






                



                


