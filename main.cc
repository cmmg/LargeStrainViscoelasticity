// For dividing the domain into cells (not necessarily the same things as elements)
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>

#include "parameters.h"

using namespace dealii;

int main() {

    std::cout << "\n-- Start\n" << std::endl;

    // Create an object that will contain the initial mesh. The dim parameter
    // decides whether it is 2D or 3D
    Triangulation<dim> triangulation;

    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);

    DoFHandler<dim> dof_handler(triangulation);
    FE_Q<dim> fe(1);

    dof_handler.distribute_dofs(fe);

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    std::ofstream output_file("solution.vtu");

    /*data_out.build_patches();*/

    data_out.write_vtu (output_file);

    std::cout << "\n-- End\n" << std::endl;

    return 0;

}
