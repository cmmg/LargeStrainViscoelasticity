import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for use over SSH
import matplotlib.pyplot as plt

import vtk
import glob

class VtuSigmaExtractor:
    def __init__(self, directory, no_of_files):
        # Initialize with the directory containing the VTU files
        self.directory = directory
        self.no_of_files = no_of_files
        self.sigma_yz_values = []

    def extract_sigma_yz(self):
        # Get a list of all .vtu files in the directory
        # vtu_files = glob.glob(f"{self.directory}/solution-*.vtu")

        vtu_files = [ f"{self.directory}/solution-{i}.vtu" for i in range(no_of_files) ]

        for vtu_file in vtu_files:
            # Read the VTU file
            # print(vtu_file)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_file)
            reader.Update()

            # Get the data from the file
            data = reader.GetOutput()
            sigma_yz_array = data.GetPointData().GetArray("sigma_yz")

            if sigma_yz_array is not None:
                # Extract the first element of the sigma_yz array
                first_sigma_yz_value = sigma_yz_array.GetTuple(0)[0]
                self.sigma_yz_values.append(first_sigma_yz_value)
            else:
                print(f"Warning: 'sigma_yz' array not found in file {vtu_file}")

        return self.sigma_yz_values

# Start extraction -------------------------------------------------------------

directory = "/home/skunda/problems/large_deformation_viscoelasticity/solution"

no_of_files = 400

extractor = VtuSigmaExtractor(directory, no_of_files)

sigma_yz_values = extractor.extract_sigma_yz()

plt.plot(sigma_yz_values)
image_path = f"{directory}/sigma_yz.png"
print(f"Image saved to {image_path}")
plt.savefig(image_path)
