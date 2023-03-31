#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>



extern "C" {
    // solve N x N linear system with a given stencil
    void eigen_pcg_solve(int N, const float * stencil, const int* offsets, int noffsets, const float *b, float * out) {
        std::vector<Eigen::Triplet<float> > triplets;
        for (int i = 0; i < N; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, stencil[0]));
            for (int j = 1; j < noffsets; ++j) {
                int idx = i + offsets[j];
                if (idx >= 0 && idx < N) {
                    triplets.push_back(Eigen::Triplet<float>(i, idx, stencil[j]));
                }
            }
        }
        Eigen::SparseMatrix<float> A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        Eigen::Map<const Eigen::VectorXf> bmap(b, N);
        Eigen::Map<Eigen::VectorXf> outmap(out, N);
        outmap = cg.solve(bmap);
    }
}