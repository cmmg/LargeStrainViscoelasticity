#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include "parameters.h"

using namespace dealii;

namespace problem {
    
}

int main() {

    std::cout << "\n-- Program start\n" << std::endl;

    Triangulation<dim> triangulation;

    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);


    std::cout << "-- Program end\n" << std::endl;

    return 0;

}
