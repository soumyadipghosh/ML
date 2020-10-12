#include <torch/torch.h>
#include <iostream>
#include "mpi.h"
#include <cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>
//#include "cwkernels.cu"
#define LTAG 2
#define RTAG 10

std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct Model : torch::nn::Module {
    Model()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};






int main(int argc, char *argv[])
{

    int file_write = (int)std::atoi(argv[1]);

    // MPI variables
    int rank, numranks, localRank = 0;
    int left, right;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Request req1, req2;
    torch::Device device = (torch::Device(torch::kCUDA, rank));
    cudaSetDevice(rank);
//std::cout << "rank: " << rank <<"; local rank: " << localRank << std::endl; 
   // ring arrangement
    if (rank == 0)
        left = numranks - 1;
    else
        left = rank - 1;

    if (rank == numranks - 1)
        right = 0;
    else
        right = rank + 1;

    // end MPI variables
  cudaStream_t s;
  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(rank));
  CUDACHECK(cudaStreamCreate(&s));
  //initializing NCCL
    // Timer variables
    auto tstart = 0.0;
    auto tend = 0.0;
/*
int *vers;
ncclGetVersion(vers);

std::cout << "version: " << *(vers) << std::endl;
*/
    // Read dataset
    std::string filename =
        "/scratch365/rschaef1/MNIST/raw";
    auto dataset =
        torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    // Distributed Random Sampler - use sequential for gradient descent
    auto data_sampler = torch::data::samplers::DistributedSequentialSampler(
        dataset.size().value(), numranks, rank, false);
    auto num_train_samples_per_pe = dataset.size().value() / numranks;

    // Generate dataloader
    int64_t batch_size = num_train_samples_per_pe;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     data_sampler, batch_size);
    /*
    int testsize = 1024;
    int  *in;
    int *out;
    CUDACHECK(cudaMalloc((void**) &in, testsize * sizeof(int)));
    CUDACHECK(cudaMalloc(i(void**) &out, testsize * sizeof(int)));
    for(int i = 0; i < testsize; ++i) {
      CUDACHECK(cudaMemSet(&(in+i), i, sizeof(int)));
    }
    shift<<<4, 256>>>(in,out,testsize);
    int *a;
    CUDACHECK(cudaMemcpy(a, out, testsize*sizeof(int), cudaMemcpyDevicetoHost));
    std::cout << "Success!" << std::endl;
    */
    // setting manual seed - CHECK WHETHER THIS MESSES UP RANDOMNESS IN SGD
    // LATER
    torch::manual_seed(0);

    auto model = std::make_shared<Model>();
    model->to(device);
    auto sz = model->named_parameters().size();
    auto param = model->named_parameters();
    // counting total number of elements in the model
    int num_elem_param = 0;
    for (int i = 0; i < sz; i++) {
        num_elem_param += param[i].value().numel();
    }
    if (rank == 0) {
        std::cout << "Number of parameters - " << sz << std::endl;
        std::cout << "Number of elements - " << num_elem_param << std::endl;
    }
 
    auto param_elem_size = param[0].value().element_size();

    // arrays for storing left and right params
    //float left_param[num_elem_param];
    //float right_param[num_elem_param];
    float *left_param;
    float *right_param;
    float *temp;
    float *left_recv;
    float *right_recv;
  CUDACHECK(cudaMallocManaged((void**)&left_param, num_elem_param*param_elem_size));
  CUDACHECK(cudaMallocManaged((void**)&right_param, num_elem_param*param_elem_size));
  //CUDACHECK(cudaMemset(left_param, 0, num_elem_param*param_elem_size));
  //CUDACHECK(cudaMemset(right_param, 0, num_elem_param*param_elem_size));
  for (int i = 0; i < num_elem_param; ++i) {
    left_param[i] = i;
    right_param[i] = i;
  }
   CUDACHECK(cudaMallocManaged((void**)&temp, num_elem_param * param_elem_size));
   CUDACHECK(cudaMallocManaged((void**)&left_recv, num_elem_param * param_elem_size));    
   CUDACHECK(cudaMallocManaged((void**)&right_recv, num_elem_param * param_elem_size));

    auto learning_rate = 1e-2;
    torch::optim::SGD optimizer(model->parameters(), learning_rate);

    // File writing
    char name[30], pe_str[3];
    std::ofstream fp;
    sprintf(pe_str, "%d", rank);
    strcpy(name, "values");
    strcat(name, pe_str);
    strcat(name, ".txt");

    if (file_write == 1) {
    	std::cout << "File writing is on" << std::endl;
        fp.open(name);
    }
    // end file writing

    // Number of epochs
    auto num_epochs = 10; //250;
std::cout << "device: " << device << std::endl;
    // start timer
    tstart = MPI_Wtime();
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        int num_correct = 0;
        for (auto &batch : *data_loader) {
            auto ip = batch.data.cuda();
            auto op = batch.target.squeeze().cuda();

            // convert to required formats
            ip = ip.to(torch::kF32).to(device);
            op = op.to(torch::kLong).to(device);
	    //ip.to(device);
	    //op.to(device);
//std::cout << "ip.to(device)" << std::endl;
            // Reset gradients
            model->zero_grad();

            // Execute forward pass
            auto prediction = model->forward(ip);

            // Compute cross entropy loss
            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);
//	    std::cout << "Made it here!" << std::endl;
            // Print loss
            if (epoch % 1 == 0 && file_write == 1) {
                fp << epoch << ", " << loss.item<float>() << std::endl;
            }

            // Backpropagation
            loss.backward();

            int disp = 0;  // displacement of left and right params
            for (auto i = 0; i < sz; i++) {
                // getting dimensions of tensor

                int num_dim = param[i].value().dim();
                std::vector<int64_t> dim_array;
                for (int j = 0; j < num_dim; j++) {
                    dim_array.push_back(param[i].value().size(j));
                }
		
                // flattening the tensor and copying it to a 1-D vector
                auto flat = torch::flatten(param[i].value()).cuda().to(device);
//		float *temp;
		auto holder = torch::numel(flat);
//	        CUDACHECK(cudaMalloc(&temp, holder * param_elem_size));
                //auto temp = (float *)calloc(flat.numel(),
                //                            flat.numel() * param_elem_size);
                for (int j = 0; j < holder; j++) {
                   temp[j] = flat[j].item<float>();
		   //CUDACHECK(cudaMemset(temp + j, flat[j].item<float>(), sizeof(float)));
                }
	
		CUDACHECK(cudaDeviceSynchronize());


                MPI_Issend(temp, flat.numel(), MPI_FLOAT, left, RTAG,
                           MPI_COMM_WORLD, &req1);

                // send parameters to right
                MPI_Issend(temp, flat.numel(), MPI_FLOAT, right, LTAG,
                           MPI_COMM_WORLD, &req2);

                // receive from left
                MPI_Recv((left_param + disp), flat.numel(), MPI_FLOAT, left,
                         LTAG, MPI_COMM_WORLD, &status);

                // receive from right
                MPI_Recv((right_param + disp), flat.numel(), MPI_FLOAT, right,
                         RTAG, MPI_COMM_WORLD, &status);

/*	

                // send parameters to left
                NCCLCHECK(ncclGroupStart());
                NCCLCHECK(ncclSend(temp, flat.numel(), ncclFloat, left,
                           comm, s));
                NCCLCHECK(ncclRecv((left_param + disp), flat.numel(), ncclFloat, left,
                             comm, s));
		 NCCLCHECK(ncclGroupEnd());
                // send parameters to right
                NCCLCHECK(ncclGroupStart());
                NCCLCHECK(ncclSend(temp, flat.numel(), ncclFloat, right,
                           comm, s));
                NCCLCHECK(ncclRecv((right_param + disp), flat.numel(), ncclFloat, right,
                          comm, s));
		NCCLCHECK(ncclGroupEnd());
		std::cout << "*Data Transfer Succesful" << std::endl;
*/
                MPI_Wait(&req1, &status);
                MPI_Wait(&req2, &status);
CUDACHECK(cudaDeviceSynchronize());
//cudaPointerAttributes attributes;
//cudaPointerAttributes attributes2;
                // unpack 1-D vector form corresponding displacement and form
                // tensor
               //std::cout << "flat.numel(): " << flat.numel() << std::endl;
	       //std::cout << "num_elem_param: " << num_elem_param << std::endl;
  //            float *left_recv, *right_recv;
//	      CUDACHECK(cudaMalloc(&left_recv, holder * param_elem_size));
                for (int j = 0; j < holder; ++j) {
                //cudaPointerGetAttributes(&attributes, left_recv); 
		//cudaPointerGetAttributes(&attributes2, left_param); 
		 //*(left_recv + j) = *(left_param + disp + j);
		 left_recv[j] = left_param[disp + j];
		   //CUDACHECK(cudaMemset(right_recv + j, **(left_param+disp+j), param_elem_size));
		  //CUDACHECK(cudaMemcpy(left_param+disp+j,left_recv+j,param_elem_size,cudaMemcpyDefault));
                  
		}

//		CUDACHECK(cudaMalloc(&right_recv, holder * param_elem_size));
                for (int j = 0; j < holder; ++j) {
                   right_recv[j] = right_param[disp+j];
                 // CUDACHECK(cudaMemset(right_recv + j, *(right_param+disp+j), param_elem_size));
                 // CUDACHECK(cudaMemcpy(right_param+disp+j,right_recv+j,param_elem_size,cudaMemcpyDefault));
                }	
		//   torch::from_blob(left_recv, dim_array, torch::kFloat)
		//	.clone().to(device);
             /*   auto left_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                // fp << "left - " << std::endl;
                for (int j = 0; j < flat.numel(); j++) {
                    *(left_recv + j) = *(left_param + disp + j);
                }*/

  /*              torch::Tensor left_tensor = torch::from_blob(left_recv, dim_array, torch::kFloat)
                        .cuda().to(device);
                left_tensor.clone();
*/
 auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, rank); 

                torch::Tensor left_tensor = torch::from_blob(left_recv, dim_array, options).clone();
                      //  .clone().cuda();
               // left_tensor.to(device);

	    /*
                auto right_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++) {
                    *(right_recv + j) = *(right_param + disp + j);
                }*/
/*
                torch::Tensor right_tensor =
                    torch::from_blob(right_recv, dim_array, torch::kFloat)
                       .cuda().to(device);
		right_tensor.clone();
*/
                torch::Tensor right_tensor =
                    torch::from_blob(right_recv, dim_array, options).clone();
                 //      .clone().cuda();
                //right_tensor.to(device);

                // average gradients
                param[i].value().data().add_(left_tensor.data());
                param[i].value().data().add_(right_tensor.data());
                param[i].value().data().div_(3);

                // updating displacement
                disp = disp + flat.numel();

                // freeing temp arrays*/
            }

            // Update parameters
            optimizer.step();

            // Accuracy
            auto guess = prediction.argmax(1);
            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
        }  // end batch loader

        auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;

        std::cout << epoch << ", " << accuracy << std::endl;

        /*
        // Printing parameters to file
        auto param0 = torch::norm(param[0].value()).item<float>();
        auto param1 = torch::norm(param[1].value()).item<float>();
        auto param2 = torch::norm(param[2].value()).item<float>();
        auto param3 = torch::norm(param[3].value()).item<float>();
        if (file_write == 1)
            fp << epoch << ", " << param0 << ", " << param1 << ", " << param2
               << ", " << param3 << std::endl;
        */

    }  // end epochs
                CUDACHECK(cudaFree(temp));
                CUDACHECK(cudaFree(left_recv));
                CUDACHECK(cudaFree(right_recv));

    // end timer
    tend = MPI_Wtime();
    if (rank == 0)
        std::cout << "Training time - " << (tend - tstart) << std::endl;

    if (file_write == 1) fp.close();

    // Averaging learnt model - relevant only for rank 0
    for (int i = 0; i < sz; i++) {
        MPI_Allreduce(MPI_IN_PLACE, param[i].value().data_ptr(),
                      param[i].value().numel(),
                      mpiDatatype.at(param[i].value().scalar_type()), MPI_SUM,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            param[i].value().data() = param[i].value().data() / numranks;
        }
    }

    // Testing only in rank 0
    if (rank == 0) {
        auto test_dataset =
            torch::data::datasets::MNIST(
                filename, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());

        auto num_test_samples = test_dataset.size().value();
        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset), num_test_samples);

        model->eval();

        int num_correct = 0;

        for (auto &batch : *test_loader) {
            auto ip = batch.data.cuda();
            auto op = batch.target.squeeze().cuda();

            // convert to required format
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);
	    ip.to(device);
	    op.to(device);
            auto prediction = model->forward(ip);

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            std::cout << "Test loss - " << loss.item<float>() << " "
                      << std::endl;

            auto guess = prediction.argmax(1);

            // std::cout << "Prediction: " << std::endl << prediction <<
            // std::endl;

            /*
            std::cout << "Output  Guess" << std::endl;
            for(auto i = 0; i < num_test_samples; i++)
            {
                std::cout << op[i].item<int64_t>() << "  " <<
            guess[i].item<int64_t>() << std::endl;
            }
            */

            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

        }  // end test loader

        std::cout << "Num correct - " << num_correct << std::endl;
        std::cout << "Test Accuracy - "
                  << 100.0 * num_correct / num_test_samples << std::endl;
    }  // end rank 0
  //finalizing NCCL
    MPI_Finalize();
}
