import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend of matplotlib for use over SSH
import matplotlib.pyplot as plt

# This program assumes that the output from dealii is in vtk, vtu etc format
import vtk

class VtuSigmaExtractor:
    def __init__(self, directory, no_of_files):
        # Initialize with the directory containing the VTU files
        self.directory = directory
        self.no_of_files = no_of_files
        self.sigma_values = []

    def extract_sigma_yz(self):

        # It is a ssumed that there is a folder within the simulation directory
        # called "solution" that contains all the relevant solution files
        vtu_files = [ f"{self.directory}/solution-{i}.vtu" for i in range(no_of_files) ]

        for vtu_file in vtu_files:
            # Read the VTU file
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_file)
            reader.Update()

            # Get the data from the file
            data = reader.GetOutput()
            sigma_array = data.GetPointData().GetArray("sigma_yz")

            if sigma_array is not None:
                # Extract the first element of the sigma_yz array
                first_sigma_value = sigma_array.GetTuple(0)[0]
                self.sigma_values.append(first_sigma_value)
            else:
                print(f"Warning: 'sigma_yz' array not found in file {vtu_file}")

        return self.sigma_values

# Set parameters ---------------------------------------------------------------

directory = "/home/skunda/problems/large_deformation_viscoelasticity/solution"

no_of_files = 500

# Start extraction -------------------------------------------------------------

extractor = VtuSigmaExtractor(directory, no_of_files)

sigma_values = extractor.extract_sigma_yz()

plt.plot(sigma_values)
image_path = f"{directory}/sigma_yz.png"
print(f"Image saved to {image_path}")
plt.savefig(image_path)
