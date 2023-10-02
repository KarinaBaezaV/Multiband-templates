import numpy as np
from os import path

class template_stack:
    
    def __init__(self, rr_type:str, path_to_data:str) -> None:
        self.rr_type = rr_type
        self.path_to_data = path_to_data
        self.all_data = self.loader()
        self.ID, self.Band, self.Period, self.Coefficients = self.all_data

    def loader(self):
        elements = ["ID", "Band", "Period", "Coefficients"]
        array_collection = np.load(path.join(self.path_to_data, f"fourier_RR{self.rr_type}_DECam.npz"), allow_pickle = True)

        return([array_collection[x] for x in elements])

class single_template:

    def __init__(self, template_stack:template_stack, ogle_number:int, band:int) -> None:
        self.name = f"OGLE-BLG-RRLYR-{ogle_number:05d}"
        self.band = band
        selection_mask = (template_stack.ID == ogle_number) & (template_stack.Band == band)
        # Add 'not found" check
        self.period = template_stack.Period[selection_mask][0]
        self.coefficients = template_stack.Coefficients[selection_mask][0]
        self.n_terms = (len(self.coefficients) - 1) // 2
    

    def evaluate_in_phase(self, time_array):
        result = np.zeros(len(time_array), dtype = np.float64) + self.coefficients[0]
        for i in range(1, self.n_terms + 1):
            argument = 2*np.pi*i*time_array# /self.period
            sines = self.coefficients[i] * np.sin(argument)
            cosines = self.coefficients[self.n_terms + i] * np.cos(argument)
            print(i, i+self.n_terms)
            result += (sines+cosines)
        return(result)
