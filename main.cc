// For dividing the domain into cells (not necessarily the same things as elements)
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
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
        void output_results();

    private:
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        DataOut<dim> data_out;

};

template <int dim>
Problem<dim>::Problem () {
    GridGenerator::hyper_cube(triangulation);
}

template <int dim>
void Problem<dim>::run () {
    setup_system();
    output_results();
}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "Setting up" << std::endl;

    triangulation.refine_global(2);

    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;

    FE_Q<dim> fe(1); 
    dof_handler.initialize(triangulation, fe);
    dof_handler.distribute_dofs(fe);

    std::cout << "Set up complete" << std::endl;
}

template <int dim>
void Problem<dim>::output_results () {
    
    data_out.attach_dof_handler(dof_handler);
    std::ofstream output_file("solution.vtu");
    data_out.build_patches();
    data_out.write_vtu (output_file);

    std::cout << "Results written" << std::endl;

}

int main() {

    std::cout << "\n-- Start\n" << std::endl;

    // Create the problem
    Problem<3> problem;

    // Solve the problem
    problem.run();

    std::cout << "\n-- End" << std::endl;

    return 0;

}
