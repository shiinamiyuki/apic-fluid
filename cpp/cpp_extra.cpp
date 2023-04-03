#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <iostream>
#include <igl/opengl/glfw/Viewer.h>

// Eigen::MatrixXd V;
// Eigen::MatrixXi F;

// int main(int argc, char *argv[])
// {
//   // Load a mesh in OFF format
//   igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);

//   // Plot the mesh
//   igl::opengl::glfw::Viewer viewer;
//   viewer.data().set_mesh(V, F);
//   viewer.launch();
// }
struct Viewer
{
    igl::opengl::glfw::Viewer inner;
    int particle_count;
    Eigen::MatrixXd P;
    Viewer(int particle_count) : particle_count(particle_count)
    {
        P.resize(particle_count, 3);
    }
};
extern "C"
{
    void *launch_viewer(size_t particle_count)
    {
        auto viewer = new Viewer(particle_count);
        return viewer;
    }
    void viewer_set_points(void *viewer_, const float *points)
    {
        auto viewer = (Viewer *)viewer_;
        auto &P = viewer->P;
        for (int i = 0; i < viewer->particle_count; i++)
        {
            P.row(i) = Eigen::RowVector3d(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
        }
        viewer->inner.data().set_points(P, Eigen::RowVector3d(1, 1, 1));
    }
    void destory_viewer(void *viewer_)
    {
        auto viewer = (Viewer *)viewer_;
        delete viewer;
    }
    // solve N x N linear system with a given stencil
    void eigen_pcg_solve(int nx, int ny, int nz, const float *stencil, const int *offsets, int noffsets, const float *b, float *out)
    {
        std::vector<Eigen::Triplet<float>> triplets;
        auto N = nx * ny * nz;
        for (int z = 0; z < nz; ++z)
        {
            for (int y = 0; y < ny; ++y)
            {
                for (int x = 0; x < nx; ++x)
                {
                    int i = x + y * nx + z * nx * ny;

                    triplets.push_back(Eigen::Triplet<float>(i, i, stencil[0]));
                    for (int j = 1; j < noffsets; ++j)
                    {
                        auto off_x = offsets[j * 3 + 0];
                        auto off_y = offsets[j * 3 + 1];
                        auto off_z = offsets[j * 3 + 2];
                        // fprintf(stderr, "off_x: %d, off_y: %d, off_z: %d\n", off_x, off_y, off_z);
                        int idx = i + off_x + off_y * nx + off_z * nx * ny;
                        if (x + off_x >= 0 && x + off_x < nx &&
                            y + off_y >= 0 && y + off_y < ny &&
                            z + off_z >= 0 && z + off_z < nz)
                        {
                            triplets.push_back(Eigen::Triplet<float>(i, idx, stencil[j]));
                        }
                    }
                }
            }
        }
        Eigen::SparseMatrix<float> A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
        cg.compute(A);
        Eigen::Map<const Eigen::VectorXf> bmap(b, N);
        Eigen::Map<Eigen::VectorXf> outmap(out, N);
        outmap = cg.solve(bmap);

        // std::cout << Eigen::MatrixXf(A) << std::endl;
        // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> solver2;
        // solver2.compute(A);
        // Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> solver;
        // solver.compute(A);
        // Eigen::Map<const Eigen::VectorXf> bmap(b, N);
        // Eigen::Map<Eigen::VectorXf> outmap(out, N);
        // std::cout <<  bmap << std::endl;
        // std::cout << "solution:" << std::endl;
        // outmap = solver.solve(bmap);
        //  std::cout <<  outmap << std::endl;

        // Eigen::VectorXf out2 = solver2.solve(bmap);
        // std::cout << "solution2:" << std::endl;
        //  std::cout <<  out2 << std::endl;
        // Eigen::VectorXf residual = (A* outmap) - bmap;
        // std::cout << "residual:" << std::endl;
        // std::cout <<  residual << std::endl;

        //  Eigen::VectorXf residual2 = (A* out2) - bmap;
        // std::cout << "residual2:" << std::endl;
        // std::cout <<  residual2 << std::endl;
        printf("iterations = %ld\n", cg.iterations());
    }
}