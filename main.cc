// For dividing the domain into cells (not necessarily the same things as elements)
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>

#include "parameters.h"

using namespace dealii;

template <int dim>
class Problem {
    public:
        Problem();
        void run();

    private:
        void setup_system();

    private:
        Triangulation<dim> triangulation;
};

template <int dim>
Problem<dim>::Problem () {
    GridGenerator::hyper_cube(triangulation);
}

template <int dim>
void Problem<dim>::run () {
    setup_system();
}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "Setting up" << std::endl;

    triangulation.refine_global(2);

    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;
}

int main() {

    std::cout << "\n-- Start\n" << std::endl;

    Problem<3> problem;

    problem.run();

    /*DoFHandler<dim> dof_handler;*/
    /*FE_Q<dim> fe(1);*/
    /**/
    /**/
    /*// Setup system*/
    /*dof_handler.initialize(triangulation, fe);*/
    /*dof_handler.distribute_dofs(fe);*/
    /**/
    /*DataOut<dim> data_out;*/
    /*data_out.attach_dof_handler(dof_handler);*/
    /*std::ofstream output_file("solution.vtu");*/
    /*data_out.build_patches();*/
    /*data_out.write_vtu (output_file);*/

    std::cout << "\n-- End" << std::endl;

    return 0;

}
