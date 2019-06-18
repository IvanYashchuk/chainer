#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        Array out_contiguous = AsContiguous(out);

        auto cholesky_impl = [&](auto pt, auto bufsize_func, auto solver_func) {
            CHAINERX_ASSERT(a.dtype() == out_contiguous.dtype());

            using T = typename decltype(pt)::type;

            // Note that cuSOLVER uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use cuSOLVER to compute an upper triangular matrix U = cholesky(A).
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            // compute workspace size and prepare workspace
            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
            int work_size = 0;
            const int N = a.shape()[0];
            device_internals.cusolverdn_handle().Call(bufsize_func, uplo, N, out_ptr, N, &work_size);

            // POTRF execution
            Array work = Empty(Shape({work_size}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            int* devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));
            device_internals.cusolverdn_handle().Call(solver_func, uplo, N, out_ptr, N, work_ptr, work_size, devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull potrf (Cholesky) execution. Info = ", devInfo_h};
            }
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by Cholesky decomposition"};
                break;
            case Dtype::kFloat32:
                cholesky_impl(PrimitiveType<float>{}, cusolverDnSpotrf_bufferSize, cusolverDnSpotrf);
                break;
            case Dtype::kFloat64:
                cholesky_impl(PrimitiveType<double>{}, cusolverDnDpotrf_bufferSize, cusolverDnDpotrf);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CholeskyKernel, CudaCholeskyKernel);

}  // namespace cuda
}  // namespace chainerx
