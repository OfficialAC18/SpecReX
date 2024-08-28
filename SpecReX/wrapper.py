from SpecReX.explanation  import summarise, explanation_wrapper
from SpecReX.spectral_explanations import fixed_beam_search
from SpecReX.model_funcs import Shape, pred_fn_wrapper
from SpecReX.visualisation import spectra_ranking_plot
from SpecReX.distributions import str2distribution

from typing import List


import numpy as np
from numpy.typing import NDArray



class SReX:
    def __init__(self, model = None, pred_fn = None, top_predictions = 1):

        #The core stuff
        self.resp = None
        self.exps = None
        self.mutants = None
        self.pred_fn = pred_fn if model is None else pred_fn_wrapper(model, top_predictions)

        #Test the there is a prediction function
        assert self.pred_fn is not None, "No prediction function has been passed, either pass a prediction func or a Pytorch/ONNX model"

        #Summaries
        self.pmax = None
        self.pmax_pos = None
        self.pmean = None
        self.pstd_dev = None
        self.pmedian = None

        #Stats
        self.passing: int = 0
        self.failing: int = 0
        self.depth_reached: int = 0
        self.avg_box_size: float = 0 

    def calc_responsibility(self, spectra: NDArray, wn: NDArray,
                            iters: int = 50, seed: int = 42,
                            distribution: str = 'uniform', distribution_args: int = 0.7,
                            tree_depth: int = 10, min_box_size: int = 25,
                            min_work: int = 0.2, interp_method: str = 'linear',
                            weighted: bool = False, search_limit: int = 200,
                            bounding_box: List[int] = None, total_restart_attempts: int = 5,
                            targets: int = None, verbose: bool = False, return_mutant_iters: int = -1):
        '''
        Calculates the responsibility values for the given input array
        and calculate the summary statistics of the responsibility map.

        Parameters
        ----------
        spectra : np.NDarray
            The spectra for which you'd like to calculate the responsibility.
            Should be of the shape (1,len), (len,1), (len,) depending on model input.

        wn : np.NDarray
            The corresponding wavenumber for the input spectra, should be of the same shape
            as the input spectra.

        iters : (int, optional)
            The number of iterations over which we calculate and average responsibility values. (Default: 50)

        seed : (int, optional)
            Seed for reproducibility of responsibility calculations. (Default: 42)

        distribution : (str, optional)
            The sampling distribution for mutant generation (Uniform, Binomial). (Default: Uniform)
        
        distirbution_args : (int, optional)
            The distribution args for the binomial distribution (Probability val). (Default: 0.7)
        
        tree_depth : (int, optional)
            The maximum depth to which the tree will be explored. (Default: 10)
        
        min_box_size : (int, optional)
            The minimum size of the mutant that will be accepted. (Default: 5)
        
        min_work : (int, optional)
            The minmum number of examples to be considered when calculating responsibility. (Default: 0.2)

        interp_method : (str, optional)
            The interpolation method used for generating mutants (linear, cubic). (Default: linear)
        
        weighted : (bool, optional)
            Determine if responsibility calculations are to be weighted with model confidence. (Default: False)
        
        search_limit : (int, optional)
            Total number of mutants explored before terminating calculations. (Default: 200)
        
        bounding_box : (List[int], optional)
            The regions of the input spectra that is considered, provide list of the form [start,end]. By default, the entire spectra 
            is considered. (Default : None)
        
        total_restart_attempts : (int, optional)
            Number of times responsibility calculations are restarted in a given iteration due to not reaching minimum work. (Default: 5)

        targets : (int, optional)
            The class(es) for which we want to attain the explanation. By default, SpecReX finds the explanation of the class that the model deems
            the input to be in. (Default : None)
        
        verbose : (bool, optional)
            Get a detailed description of each iteration. (Default: False)
        
        return_mutants_iter : (int, optional)
            Return all the mutants at a particular iteration of the responsibility calculation. (Default: -1)



        Returns:
        resp (NDArray)
        pmax (float)
        pmax_pos (int)
        pmean (float)
        pstd_dev (float)
        pmedian (int)
        extracted_mutants (List[NDArray])
        
        '''

        assert spectra.shape == wn.shape, "Spectra and Wavenumber are of different shapes"

        #Convert to 32 bit
        spectra = spectra.astype('float32')
        wn = wn.astype('float32')

        #Batching for meeting requirements of model
        self.spectra = spectra = np.expand_dims(spectra, axis = 0)
        self.wn = wn = np.expand_dims(wn, axis = 0)

        self.shape = spec_shape = Shape(spectra)
        self.interp_method = interp_method

        #Calculate the responsibility
        self.resp, self.targets, self.mutants = explanation_wrapper(
            prediction_func=self.pred_fn,
            spec_array=spectra,
            wn_array=wn,
            spec_shape=spec_shape,
            iters=iters,
            distribution=str2distribution(distribution),
            distribution_args=distribution_args,
            search_limit=search_limit,
            tree_depth=tree_depth,
            weighted=weighted,
            min_box_size=min_box_size,
            interp_method=interp_method,
            min_work=min_work,
            total_restart_attempts=total_restart_attempts,
            seed=seed,
            bounding_box=bounding_box,
            verbose = verbose,
            targets=targets,
            return_mutant_iters = return_mutant_iters,
        )

        #Calculate the summaries of the responsibility landscape
        self.pmax, self.pmax_pos, self.pmean, self.pstd_dev, self.pmedian = summarise(self.resp)
    
        return self.resp, self.pmax, self.pmax_pos, self.pmean, self.pstd_dev, self.pmedian
    

    def explanations(self, beam_size: int = 20,
                     beam_engulf_window: int = 10, beam_eta: int = 1,
                     responsibility_similarity: float = 0.9, maxima_scaling_factor: float = 0.5,
                     multiple: bool = True):
        
        '''
        Extract Explanations from the responsibility landscape of the input data, Need to calculate responsibility before 
        extracting the explanations.
        
        Parameters
        ----------
        beam_size (int, optional): The width of the region that is explored from a peak in the responsibility landscape, the region explored is
        of the form (peak - beam_size, peak + beam_size). (Default: 20)

        beam_engulf_window (int, optional): Minimum distance between potential explanation candidates. (Default: 10)

        beam_eta (int, optional): width by which we decrease the explanation in each iteration to get minimal explanations,
        the region explored will be of the form (start + iter*beam_eta, stop - iter*beam_eta)

        responsibility_similarity (float, optional): Similarity threshold for grouping peaks in order to test for conjunctive explanations (Default: 0.9)

        maxima_scaling_factor (float, optional): Scaling factor compared to previous highest maxima, if peak is below scaling factor, considered as noise 
        and not explored (Default: 0.5)

        multiple (bool, optional): Extract a single or multiple explanations from the input. (Default: True)

        Return
        ------

        exps (dict[tuple[int],tuple[int]])

        '''

        assert self.resp is not None, "Please calculate the responsbility of the spectra first via SpecReX.calc_responsibility"

        self.exps = fixed_beam_search(
            spec_array=self.spectra,
            wn_array=self.wn,
            prediction_func=self.pred_fn,
            pos_ranking=self.resp,
            interp_method=self.interp_method,
            beam_size=beam_size,
            beam_engulf_window=beam_engulf_window,
            beam_eta=beam_eta,
            responsibility_similarity=responsibility_similarity,
            maxima_scaling_factor=maxima_scaling_factor,
            multiple=multiple,
            target_class=self.targets
        )

        return self.exps
    

    def visualise(self, path = None):
        '''
        Visualises the explanations in an interpretable manner, responsbility and explanations need to be calculate before 
        visualising hte results.

        Parameters
        ----------
        path (str or Path, optional) : Path to where the visualisation will be saved. (Default None)


        Return
        ------
        None
        '''

        assert self.resp is not None, "Responsibility is not calculated, please calculate it using SpecReX.calc_responsibility"
        assert self.exps is not None, "Explanations have not been extracted, please extract using SpecReX.explanations"

        spectra_ranking_plot(
            destination=path,
            spectra = self.spectra,
            wn = self.wn,
            ranking = self.resp,
            explanations =  self.exps
        )



    