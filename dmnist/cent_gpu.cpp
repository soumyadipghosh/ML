#include "mpi.h"
#include "nccl.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdint.h>
#include <torch/torch.h>
#include <unistd.h>

std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static uint64_t getHostHash(const char *string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

// Define a new Module.
struct Model : torch::nn::Module {
  Model() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::relu(fc2->forward(x));
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main(int argc, char *argv[]) {
  int numranks, rank, localRank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  // cudaSetDevice(rank);

  torch::Device device = (torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA not available. Training on CPU." << std::endl;
  }
  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[numranks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < numranks; p++) {
    if (p == rank)
      break;
    if (hostHashs[p] == hostHashs[rank])
      localRank++;
  } // torch::Device device = (torch::kCPU);

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;
  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (rank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&s));
  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, numranks, id, rank));

  //  CUDACHECK(cudaSetDevice(rank));
  int devices = torch::cuda::device_count();
  std::cout << "devices: " << devices << std::endl;
  // std::cout << device << std::endl;

  // Timer variables
  auto tstart = 0.0;
  auto tend = 0.0;
  c10::cuda::CUDACachingAllocator::emptyCache();
  // TRAINING
  // Read train dataset
  std::string filename =
      "/afs/crc.nd.edu/user/r/rschaef1/Public/tryingout/MNIST/raw";
  auto train_dataset =
      torch::data::datasets::MNIST(filename)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  // added here, move the dataset to the device

  // Distributed Random Sampler
  // Look hard at this: what can I use instead of rank?

  auto data_sampler = torch::data::samplers::DistributedRandomSampler(
      train_dataset.size().value(), numranks, rank, false);

  auto num_train_samples_per_pe = train_dataset.size().value() / numranks;

  // Generate dataloader
  auto batch_size = num_train_samples_per_pe; // for now, since the dataset is
                                              // on the GPU
  auto data_loader = torch::data::make_data_loader(
      std::move(train_dataset), data_sampler,
      torch::data::DataLoaderOptions().batch_size(batch_size));

  // setting manual seed - CHECK WHETHER THIS MESSES UP RANDOMNESS IN SGD
  // LATER
  torch::manual_seed(0);

  auto model = std::make_shared<Model>();
  // cudaSetDevice(rank);
  model->to(device);
  // data_sampler->to(device);
  // data_loader->to(device);

  auto learning_rate = 1e-2;

  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  // File writing
  int file_write = 0;
  char name[30], pe_str[3];
  std::ofstream fp;
  sprintf(pe_str, "%d", rank);
  strcpy(name, "values");
  strcat(name, pe_str);
  strcat(name, ".txt");

  if (file_write == 1) {
    fp.open(name);
  }
  // end file writing
  // Number of epochs
  size_t num_epochs = 100;

  auto sz = model->named_parameters().size();
  auto param = model->named_parameters();
  int num_elem_param = 0;
  for (int i = 0; i < sz; i++) {
    num_elem_param += param[i].value().numel();
  }
  if (rank == 0) {
    std::cout << "Number of parameters - " << sz << std::endl;
    std::cout << "Number of elements - " << num_elem_param << std::endl;
  }
  // get the size of each element (for communication purposes)
  // start timer
  tstart = MPI_Wtime();

  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    size_t num_correct = 0;

    for (auto &batch : *data_loader) {
      // cudaMalloc(ip, batch.data.size() * sizeof(float));
      auto ip = batch.data.cuda();
      auto op = batch.target.squeeze().cuda();
      // std::cout <<"rank " << rank << " ip - " <<  ip[1] << std::endl;
      // std::cout << "ip type: " << typeid(ip).name() << std::endl;
      // convert to required formats
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);
      //        cudaSetDevice(rank);
      ip.to(device);
      op.to(device);

      // Reset gradients
      model->zero_grad();

      // Execute forward pass
      auto prediction = model->forward(ip).to(device);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      // Print loss
      if (epoch % 1 == 0) {
        // fp << "Output at epoch " << epoch << " = " << loss.item<float>() <<
        // std::endl;
        fp << epoch << ", " << loss.item<float>() << std::endl;
      }

      // Backpropagation
      loss.backward();
      // testing
      for (auto &param : model->named_parameters()) {

        NCCLCHECK(ncclAllReduce(
            param.value().grad().data_ptr(), param.value().grad().data_ptr(),
            param.value().grad().numel(), ncclFloat, ncclSum, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        param.value().grad().data() = param.value().grad().data() / numranks;
        // CUDACHECK(cudaFree(sendbuff));
        // CUDACHECK(cudaFree(recvbuff));
        /*              NCCLCHECK(ncclGroupStart());
                      for (int i = 0; i < nDev; ++i) {
                          CUDACHECK(cudaStreamCreate(s+i));
                      NCCLCHECK(ncclAllReduce(param.value().grad().data_ptr(),
           param.value().grad().data_ptr(), param.value().grad().numel(),
           ncclFloat, ncclSum, comms[i], s[i])); param.value().grad().data() =
           param.value().grad().data()/numranks;
                        }
                        for (int i = 0; i < nDev; ++i) {
                            CUDACHECK(cudaSetDevice(i));
                             CUDACHECK(cudaStreamSynchronize(s[i]));
                        }*/
      }
      optimizer.step();

      auto guess = prediction.argmax(1);
      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
      // c10::cuda::CUDACachingAllocator::emptyCache();
    } // end batch loader

    auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;
    // if (rank == 0) {
    std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - "
              << accuracy << std::endl;

  } // end epoch

  // end timer
  tend = MPI_Wtime();
  if (rank == 0) {
    std::cout << "Training time - " << (tend - tstart) << std::endl;
  }

  if (file_write == 1)
    fp.close();
  // device = torch::kCPU;
  // TESTING ONLY IN RANK 0
  if (rank == 0) {
    auto test_dataset =
        torch::data::datasets::MNIST(filename,
                                     torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto num_test_samples = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset),
                                                     num_test_samples);

    model->eval(); // enable eval mode to prevent backprop

    size_t num_correct = 0;

    for (auto &batch : *test_loader) {
      auto ip = batch.data.to(device);
      auto op = batch.target.squeeze().to(device);

      // convert to required format
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      auto prediction = model->forward(ip);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      std::cout << "Test loss - " << loss.item<float>() << std::endl;

      auto guess = prediction.argmax(1);

      /*
      std::cout << "Prediction: " << std::endl << prediction << std::endl;
      std::cout << "Output  Guess" << std::endl;
      for(auto i = 0; i < num_test_samples; i++)
      {
          std::cout << op[i].item<int64_t>() << "  " << guess[i].item<int64_t>()
      << std::endl;
      }
      */

      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

    } // end test loader

    std::cout << "Num correct - " << num_correct << std::endl;
    std::cout << "Test Accuracy - " << 100.0 * num_correct / num_test_samples
              << std::endl;
  } // end rank 0
  ncclCommDestroy(comm);

  MPI_Finalize();
}
