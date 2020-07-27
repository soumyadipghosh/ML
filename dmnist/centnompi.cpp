#include <torch/torch.h>
#include <iostream>
#include "mpi.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
// include libtorch/mpi

// DOUBLE CHECK THIS LINE --> I think this is remapping of mpi variable names?
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
        {at::kByte, MPI_UNSIGNED_CHAR},
        {at::kChar, MPI_CHAR},
        {at::kDouble, MPI_DOUBLE},
        {at::kFloat, MPI_FLOAT},
        {at::kInt, MPI_INT},
        {at::kLong, MPI_LONG},
        {at::kShort, MPI_SHORT},
};

// Define a new Module.
// Creating the network
struct Model : torch::nn::Module {
    Model()
    {
        // Construct and register two Linear submodules.
        // Starting with a 784 (28*28) layer, reducing to 10 (1-10)
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    // Implement the Net's algorithm.
    // defining forward propogation
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        // reshaping the 28*28 image into a 784*1 tensor
        // relu is the activation function
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    // DOUBLE CHECK THIS LINE
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// main function
int main(int argc, char *argv[])
{

  //MPI variables
  int rank, numranks;
  // Initialize MPI (Just a necessary step of C++ MPI)
  MPI_Init(&argc, &argv);
  // Comsize is the number of ranks, so we ran this with 4 ranks
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  // assigning rank 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // end MPI variables
       for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
       }
  int devices = torch::cuda::device_count();
   torch::Device device = (torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device =  torch::kCUDA;
  } else {
    std::cout << "CUDA not available. Training on CPU." << std::endl;
  }
  // Timer variables
  // necessary to check how long each step takes
  auto tstart = 0.0;
  auto tend = 0.0;

  // TRAINING
  // Read train dataset
  // may need to update file path
  std::string filename = "/afs/crc.nd.edu/user/r/rschaef1/Public/tryingout/MNIST/raw";
  // normalizing the dataset.... So I think I know what this means, making sure
  // all the features are within the same parameters? changing all the values
  // to a common scale in other words. 
  auto train_dataset = torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    
  // Move the Dataset to the GPU
  // Distributed Random Sampler --> loaded from libtorch
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(train_dataset.size().value(),
                                             numranks, rank, false);


  auto num_train_samples_per_pe = train_dataset.size().value() / numranks;

  // Generate dataloader
  auto batch_size = num_train_samples_per_pe; 
  auto data_loader = torch::data::make_data_loader
                     (std::move(train_dataset), data_sampler, batch_size);

  // setting manual seed - CHECK WHETHER THIS MESSES UP RANDOMNESS IN SGD
  // LATER
  torch::manual_seed(0);
  // define/make the mode



  // DUDE so focus on this line, here you're prolly gonna wanna create the net on the GPU
  // or just move it to each device after
  // so like a for loop for np
  auto model = std::make_shared<Model>();
  // define the learning rate
  model->to(device);
  auto learning_rate = 1e-2;
  // load the optimizer (which in this case is Stochastic Gradient Descent)
  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  // File writing
  // The best I can tell, this is where the final values of the model are stored
  // This is writing the name
  // file_write = 0 means that the file won't actually be made
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
  size_t num_epochs = 250;

  // start timer (question: are we optimizing for time, energy, or both?)
  tstart = MPI_Wtime();
  // train the model for homever many epochs
  for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
  {

      size_t num_correct = 0;
      // repeat for each batch
      for (auto& batch : *data_loader)
      {
          // ip = input, op = output
          auto ip = batch.data.cuda();
          auto op = batch.target.squeeze().cuda();

          // convert to required formats
          ip = ip.to(torch::kF32); 
          op = op.to(torch::kLong);

	  // Reset gradients
	  model->zero_grad();

	  // Execute forward pass
	  auto prediction = model->forward(ip);
          // calculate loss on the training set
          auto loss = torch::nll_loss(torch::log_softmax(prediction, 1) , op);

          //Print loss
          if (epoch % 1 == 0) {
	      //fp << "Output at epoch " << epoch << " = " << loss.item<float>() << std::endl;
              fp << epoch << ", " << loss.item<float>() << std::endl;
          }

	  // Backpropagation (pretty)
	  loss.backward();
          // Here is the allreducle algorithm, which is exectued over all of the parameters
          // to make sure each processor has the same gradients
          for (auto &param : model->named_parameters())
          {
              //fp << param.key() << "-  " << torch::norm(param.value()).item<float>() << std::endl;
              //fp << torch::norm(param.value().grad()).item<float>() << std::endl;

              MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(),
               param.value().grad().numel(),
                mpiDatatype.at(param.value().grad().scalar_type()),
                MPI_SUM, MPI_COMM_WORLD);

              param.value().grad().data() = param.value().grad().data()/numranks;

          } 

	  //Update parameters
	  optimizer.step();
          // run a test guess on the training set
          auto guess = prediction.argmax(1);
          num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
      } //end batch loader
      // calculate accuracy (but unless I'm mistaken this is still on the training set)
      auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;
      // print accuracy
      std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - " << accuracy << std::endl;
 
   }//end epoch 

   // end timer
   tend = MPI_Wtime();
   // rank is the processor number--only that prints out the time
   if (rank == 0) {
      std::cout << "Training time - " << (tend - tstart) << std::endl;
   }
   // close the file is if we were in fact making the file
   if(file_write == 1) fp.close();

   //TESTING ONLY IN RANK 0
   if(rank == 0)
   {
       //load the test dataset and normalize
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
          auto ip = batch.data.cuda();
          auto op = batch.target.squeeze().cuda();

          //convert to required format
          ip = ip.to(torch::kF32);
          op = op.to(torch::kLong);
          // setting the prediction to the forward model?? double check this one
          auto prediction = model->forward(ip);
          // calculate loss
          auto loss = torch::nll_loss(torch::log_softmax(prediction, 1) , op);
          // print loss
          std::cout << "Test loss - " << loss.item<float>() << std::endl;
          // prediction based on model
          auto guess = prediction.argmax(1);
        
          /*
          std::cout << "Prediction: " << std::endl << prediction << std::endl;
          std::cout << "Output  Guess" << std::endl;
          for(auto i = 0; i < num_test_samples; i++)
          {
              std::cout << op[i].item<int64_t>() << "  " << guess[i].item<int64_t>() << std::endl;
          }
          */
         // calculate correct predictions
          num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

       }// end test loader
        // report the test accuracy
       std::cout << "Num correct - " << num_correct << std::endl;
       std::cout << "Test Accuracy - "
                  << 100.0 * num_correct / num_test_samples << std::endl;
   }// end rank 0 

    // A necessary line at the end of the main function in a C MPI script
   MPI_Finalize();
}
