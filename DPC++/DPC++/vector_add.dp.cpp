//==============================================================
// Copyright   2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#define VECTOR_SIZE 25600

void VectorAddKernel(float* A, float* B, float* C, sycl::nd_item<3> item_ct1)
{
    A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
    B[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
    C[item_ct1.get_local_id(2)] =
        A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
}

int main() try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    float *d_A, *d_B, *d_C;
    int status;

    d_A = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    d_B = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    d_C = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);

    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, VECTOR_SIZE),
                                        sycl::range(1, 1, VECTOR_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             VectorAddKernel(d_A, d_B, d_C, item_ct1);
                         });
    });

    float Result[VECTOR_SIZE] = { };

    /*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    status = (q_ct1.memcpy(Result, d_C, VECTOR_SIZE * sizeof(float)).wait(), 0);

    sycl::free(d_A, q_ct1);
    sycl::free(d_B, q_ct1);
    sycl::free(d_C, q_ct1);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%3.0f ", Result[i]);    
    }
    printf("\n");
	
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
