// For dividing the domain into cells (not necessarily the same things as elements)
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe.h>
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
        void queries();

    private:
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        DataOut<dim> data_out;
};

template <int dim>
Problem<dim>::Problem ()  {
    GridGenerator::hyper_cube(triangulation);
}

template <int dim>
void Problem<dim>::run () {
    setup_system();
    queries();
    output_results();
}

template <int dim>
void Problem<dim>::queries () {

    std::vector<Point<dim>> all_vertices = triangulation.get_vertices();

    std::cout << "\nCoordinates of all vertices" << std::endl;
    for(auto vertex : all_vertices) {
        std::cout << " x = " << vertex[0]
                  << " y = " << vertex[1]
                  << " z = " << vertex[2]
                  << " norm = " << vertex.norm()
                  << std::endl;
    }
}

template <int dim>
void Problem<dim>::output_results () {
    
    data_out.attach_dof_handler(dof_handler);
    std::ofstream output_file("solution.vtu");
    data_out.build_patches();
    data_out.write_vtu (output_file);

}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "Setting up" << std::endl;

    /*triangulation.refine_global(2);*/

    FE_Q<dim> fe(1); 
    dof_handler.initialize(triangulation, fe);
    dof_handler.distribute_dofs(fe);

    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;
    std::cout << "No of dofs : " << dof_handler.n_dofs() << std::endl;

}
    
int main() {

    std::cout << "\n-- Simulation Started\n" << std::endl;

    Problem<3> problem;

    problem.run();

    std::cout << "\n-- Simulation Ended\n" << std::endl;

    return 0;

}
