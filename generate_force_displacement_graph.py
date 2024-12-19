import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend of matplotlib for use over SSH
import matplotlib.pyplot as plt

class Reader:

    def __init__(self):

        # self.force_displacement_data_file_name = "text_output_file.txt"
        self.force_displacement_data_file_name = "force_displacement_file.txt"
        self.parameters_file_name = "parameters.json"

        self.force        = list()
        self.displacement = list()

    def read_force_displacement_data_into_arrays(self):
        
        data = np.loadtxt(self.force_displacement_data_file_name)

        self.force        = data[:, 0];
        self.displacement = data[:, 1];

    def read_parameters(self):

        with open(self.parameters_file_name, "r") as parameters_file:
            all_parameters_data = json.load(parameters_file)

        material_parameters_data = all_parameters_data["Viscoelastic Material Parameters"]

        self.K         = material_parameters_data["Bulk Modulus"]
        self.mu0       = material_parameters_data["Shear Modulus"]
        self.sigma0    = material_parameters_data["Viscous Resistance"]
        self.gammadot0 = material_parameters_data["Dimensional Scaling Constant"]
        self.m         = material_parameters_data["Strain Rate Exponent"]

    def generate_force_displacement_graph(self):

        figure, axes = plt.subplots()

        axes.plot(self.force, self.displacement)

        axes.set_xlabel("Displacement")
        axes.set_ylabel("Force")

        # parameter_info = \
        # f"K         = {self.K}"         + "\n" + \
        # f"mu0       = {self.mu0}"       + "\n" + \
        # f"sigma0    = {self.sigma0}"    + "\n" + \
        # f"gammadot0 = {self.gammadot0}" + "\n" + \
        # f"m         = {self.m}"
        #
        # axes.text(
        #     1.05, 0.5,
        #     parameter_info,
        #     transform=axes.transAxes,
        #     fontsize=10,
        #     fontfamily='monospace',
        #     bbox=dict(facecolor='white', alpha=1.0),
        #     verticalalignment='center')
        #
        plt.tight_layout();

        plt.savefig("force_displacement.png")

# ------------------------------------------------------------------------------

reader = Reader()

reader.read_force_displacement_data_into_arrays()
# reader.read_parameters()
reader.generate_force_displacement_graph()
