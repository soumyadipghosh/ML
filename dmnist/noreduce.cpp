#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "mpi.h"
#include "nccl.h"
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
        {at::kByte, MPI_UNSIGNED_CHAR},
        {at::kChar, MPI_CHAR},
        {at::kDouble, MPI_DOUBLE},
        {at::kFloat, MPI_FLOAT},
        {at::kInt, MPI_INT},
        {at::kLong, MPI_LONG},
        {at::kShort, MPI_SHORT},
};


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
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


// Define a new Module.
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
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main(int argc, char *argv[])
{  
  ncclComm_t comms[4];
  int numranks, rank; 
  //managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 }; 
  //initializing NCCL
 // NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);  
//int rank, local_rank, local_size;
  MPI_Init(&argc, &argv);
  //MPI_Comm local_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,  MPI_INFO_NULL, &local_comm);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  //MPI_Comm_rank(local_comm, &local_rank);
  //cudaSetDevice(local_rank%local_size);
  //std::cout << "numranks: " << numranks << std::endl;
  //std::cout << "rank: "<< rank << std::endl; 
  //std::cout <<" local rank:" << local_rank << std::endl;
// necessary for later code, update this
       for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
       }
        torch::Device device = (torch::kCPU);
	if (torch::cuda::is_available()) {
  		std::cout << "CUDA is available! Training on GPU." << std::endl;
  		device = torch::kCUDA;
	} else {
		std::cout << "CUDA not available. Training on CPU." << std::endl;
	}	
//torch::Device device = (torch::kCPU);
 int devices = torch::cuda::device_count();
std::cout << "devices: " << devices << std::endl;
 //std::cout << device << std::endl; 
 //MPI variables
  //end MPI variables
//  std::cout << numranks << std::endl;
//  std::cout << rank << std::endl;  
  // Timer variables
  auto tstart = 0.0;
  auto tend = 0.0;

  //TRAINING
  //Read train dataset
  std::string filename = "/afs/crc.nd.edu/user/r/rschaef1/Public/tryingout/MNIST/raw";
  auto train_dataset = torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>()); 
  // added here, move the dataset to the device
  
  //Distributed Random Sampler
  // Look hard at this: what can I use instead of rank?
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(train_dataset.size().value(),
                                             4, 0, false);
  
  
  auto num_train_samples_per_pe = train_dataset.size().value() / 4;
  
  //Generate dataloader
  auto batch_size = num_train_samples_per_pe;// for now, since the dataset is on 
			  // the GPU
  auto data_loader = torch::data::make_data_loader
                     (std::move(train_dataset), data_sampler, torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));
   
  // setting manual seed - CHECK WHETHER THIS MESSES UP RANDOMNESS IN SGD
  // LATER
  torch::manual_seed(0);

  auto model = std::make_shared<Model>();
  for (int i = 0; i < devices; ++i) {
  cudaSetDevice(i);
  model->to(device);
  }
  //data_sampler->to(device);
  //data_loader->to(device);
   
  auto learning_rate = 1e-2;


  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  //File writing
  int file_write = 0;
  char name[30], pe_str[3];
  std::ofstream fp;
  sprintf(pe_str, "%d", rank);
  strcpy(name, "values");
  strcat(name, pe_str);
  strcat(name, ".txt");

  if(file_write == 1)
  {
    fp.open(name);
  }
  //end file writing

  // Number of epochs
  size_t num_epochs = 10;

  // start timer
  tstart = MPI_Wtime();

  for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
  {
      size_t num_correct = 0;

      for (auto& batch : *data_loader)
      {
          //cudaMalloc(ip, batch.data.size() * sizeof(float));
 	  auto ip = batch.data.cuda();
          auto op = batch.target.squeeze().cuda();
          
          //convert to required formats
          ip = ip.to(torch::kF32); 
          op = op.to(torch::kLong);
          for (int i = 0; i < devices; ++i) {
            cudaSetDevice(i);
            ip.to(device);
            op.to(device);
          }
	  //Reset gradients
	  model->zero_grad();

	  //Execute forward pass
	  auto prediction = model->forward(ip).to(device);

          auto loss = torch::nll_loss(torch::log_softmax(prediction, 1) , op);
          
          //Print loss
          if (epoch % 1 == 0) {
	      //fp << "Output at epoch " << epoch << " = " << loss.item<float>() << std::endl;
              fp << epoch << ", " << loss.item<float>() << std::endl;
          }
          
	  //Backpropagation
	  loss.backward();

          /*for (auto &param : model->named_parameters())
          {
              //fp << param.key() << "-  " << torch::norm(param.value()).item<float>() << std::endl;
              //fp << torch::norm(param.value().grad()).item<float>() << std::endl;
	      
              //MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(),
              // param.value().grad().numel(),
               // mpiDatatype.at(param.value().grad().scalar_type()),
               // MPI_SUM, MPI_COMM_WORLD);
              
              NCCLCHECK(ncclGroupStart());
              for (int i = 0; i < nDev; ++i) {
                  CUDACHECK(cudaStreamCreate(s+i));
              NCCLCHECK(ncclAllReduce(param.value().grad().data_ptr(), param.value().grad().data_ptr(), param.value().grad().numel(), ncclFloat, ncclSum,
        comms[i], s[i]));
              param.value().grad().data() = param.value().grad().data()/numranks;
                for (int i = 0; i < nDev; ++i) {
                    CUDACHECK(cudaSetDevice(i));
                     CUDACHECK(cudaStreamSynchronize(s[i]));
                }
          }*/ 

	  //Update parameters
	  optimizer.step();

          auto guess = prediction.argmax(1);
          num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
          c10::cuda::CUDACachingAllocator::emptyCache();
	} //end batch loader

      auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;
      if (rank == 0) {
        std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - " << accuracy << std::endl;
      }
   }//end epoch 
   
   // end timer
   tend = MPI_Wtime();
   if (rank == 0) {
      std::cout << "Training time - " << (tend - tstart) << std::endl;
   }

   if(file_write == 1) fp.close();
  // device = torch::kCPU;
   //TESTING ONLY IN RANK 0
  // if(rank == 0)
  // {
      auto test_dataset = torch::data::datasets::MNIST(
                filename, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());

      auto num_test_samples = test_dataset.size().value();
      auto test_loader = torch::data::make_data_loader
                     (std::move(test_dataset), num_test_samples);

      model->eval(); //enable eval mode to prevent backprop

      size_t num_correct = 0;

      for (auto& batch : *test_loader)
      {
          auto ip = batch.data.to(device);
          auto op = batch.target.squeeze().to(device);

          //convert to required format
          ip = ip.to(torch::kF32);
          op = op.to(torch::kLong);

          auto prediction = model->forward(ip);

          auto loss = torch::nll_loss(torch::log_softmax(prediction, 1) , op);

          std::cout << "Test loss - " << loss.item<float>() << std::endl;

          auto guess = prediction.argmax(1);

          /*
          std::cout << "Prediction: " << std::endl << prediction << std::endl;
          std::cout << "Output  Guess" << std::endl;
          for(auto i = 0; i < num_test_samples; i++)
          {
              std::cout << op[i].item<int64_t>() << "  " << guess[i].item<int64_t>() << std::endl;
          }
          */

          num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

       }//end test loader

       std::cout << "Num correct - " << num_correct << std::endl;
       std::cout << "Test Accuracy - "
                  << 100.0 * num_correct / num_test_samples << std::endl;
 //  }//end rank 0 

 for (int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
 
 MPI_Finalize();
}
