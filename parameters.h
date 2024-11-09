#define top_surface_speed 1      // Speed of the top surface

// Material parameters
#define mu_0 20 // Shear modulus [Pa]
#define lambda_L 1.09 // Maximum stretch [-]
#define K 800 // Bulk modulus [Pa]
#define sigma_0 25 // Strength parameter for viscous stretch rate
#define np 3 // Exponent for viscous flow rule
#define G_0 4500 // Elastic modulus for element C
#define G_infinity 600 // Elastic modulus for element E
#define eta 60000 // Viscosity for element D
#define gamma_dot_0 1e-4 // Dimensionless scaling constant
#define alpha 0.005 // For removing singularity in flow rule
#define f_1 0.8 // Incompressible fraction of volume to account for fluid
