// For dividing the domain into cells (not necessarily the same things as elements)
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>

#include "parameters.h"

using namespace dealii;

template <int dim>
class Problem {
    public:
        Problem();
        void run();

    private: // functions
        void setup_system();
        void assemble_system();
        void output_results();

    private: // variables
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        DataOut<dim> data_out;
        FESystem<dim> fe;
        Vector<double> system_rhs;
        Vector<double> solution;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
};

template <int dim>
Problem<dim>::Problem () : 
    dof_handler(triangulation),
    fe(FE_Q<dim>(1)^dim)
    {}

template <int dim>
void Problem<dim>::run () {
    setup_system();
    assemble_system();
    output_results();
}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "-- Setting up\n" << std::endl;

    GridGenerator::hyper_cube(triangulation);
    /*triangulation.refine_global(2);*/
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;
    std::cout << "No of dofs : " << dof_handler.n_dofs() << std::endl;
    std::cout << "FE system dealii name : " << fe.get_name() << std::endl;

    std::cout << "\n-- Set up complete" << std::endl;
}

template <int dim>
void Problem<dim>::assemble_system () {

    std::cout << "\n-- Assembling system\n" << std::endl;

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values |
                            update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    // ------------- Tangent modulus computation begin -------------

    // Declare and compute the Kronecker delta tensor
    Tensor<2, dim> kronecker_delta; 
    kronecker_delta = 0;
    kronecker_delta[0][0] = 1;
    kronecker_delta[1][1] = 1;
    kronecker_delta[2][2] = 1;

    // Compute the Lamè parameters
    double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    double mu     = E / (2 * (1 + nu));

    // Calculate the tangent modulus for hyperelasticity
    Tensor<4, dim> tangent_modulus;
    for(unsigned int i = 0; i < dim; i++) {
        for(unsigned int j = 0; j < dim; j++) {
            for(unsigned int k = 0; k < dim; k++) {
                for(unsigned int l = 0; l < dim; l++) {
                    tangent_modulus[i][j][k][l] = 
                        lambda * kronecker_delta[i][j] * kronecker_delta[k][l]
                        +
                        mu * (kronecker_delta[i][k] * kronecker_delta[j][l] +
                              kronecker_delta[i][l] * kronecker_delta[j][k]);
                }
            }
        }
    }

    // -------------- Tangent modulus computation end --------------

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    unsigned int n_quadrature_points = fe_values.n_quadrature_points;

    // Loop over all the cells of the triangulation
    for (const auto &cell : dof_handler.active_cell_iterators()) {

        // Initialize the fe_values object with values relevant to the current cell
        fe_values.reinit(cell);

        // Initialize the cell matrix and the right hand side vector with zeros
        cell_matrix = 0; 
        cell_rhs = 0;

        // Quadrature loop for current cell
        for (unsigned int q = 0; q < n_quadrature_points; q++) {

            for (unsigned int i = 0; i < dofs_per_cell; i++) {

                const unsigned int ci = fe_values
                                        .get_fe()
                                        .system_to_component_index(i)
                                        .first;

                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    const unsigned int cj = fe_values
                                            .get_fe()
                                            .system_to_component_index(i)
                                            .first;

                    for (unsigned int di; di < dim; di++) {
                        for (unsigned int dj; dj < dim; dj++) {
                            cell_matrix(i, j) +=
                                fe_values.shape_grad(i, q)[di] *
                                tangent_modulus[ci][di][cj][dj] *
                                fe_values.shape_grad(j, q)[dj] *
                                fe_values.JxW(q);
                        }
                    }

                }
            }
        }
    }

    std::cout << "\n-- Assembly complete" << std::endl;
}

template <int dim>
void Problem<dim>::output_results () {
    
    data_out.attach_dof_handler(dof_handler);
    std::ofstream output_file("solution.vtu");
    data_out.build_patches();
    data_out.write_vtu (output_file);

    std::cout << "\nResults written" << std::endl;
}

int main() {

    std::cout << "\n---- Simulation started\n" << std::endl;

    // Create the problem
    Problem<3> problem;

    // Solve the problem
    problem.run();

    std::cout << "\n---- Simulation ended" << std::endl;

    return 0;

}
