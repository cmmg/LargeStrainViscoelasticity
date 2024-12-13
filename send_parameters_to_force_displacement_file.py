import json

with open("parameters.json", "r") as parameters_file:
    all_parameters_data = json.load(parameters_file)

material_parameters_data = all_parameters_data["Viscoelastic Material Parameters"]

K         = material_parameters_data["K"]
mu0       = material_parameters_data["mu0"]
sigma0    = material_parameters_data["sigma0"]
gammadot0 = material_parameters_data["gammadot0"]
m         = material_parameters_data["m"]

output_string = \
        f"-------------------------------------------------" + "\n" + \
        f"Parameter list : " + "\n" + \
        f"K = {K}" + "\n" + \
        f"mu0 = {mu0}" + "\n" + \
        f"sigma0 = {sigma0}" + "\n" + \
        f"gammadot0 = {gammadot0}" + "\n" + \
        f"m = {m}" + "\n"

with open("force_displacement_file.txt", "a") as output_file:
    output_file.write(output_string)
