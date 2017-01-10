//#include <boost/thread.hpp>
//#include "boost.hpp"
//#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <unistd.h>

#include "caffe/common.hpp"
//#include "caffe/util/rng.hpp"

namespace caffe {

// Make sure each thread can have different values.
//static boost::thread_specific_ptr<Caffe> thread_instance_;

Caffe& Caffe::Get() {
  //if (!thread_instance_.get()) {
  //  thread_instance_.reset(new Caffe());
  //}
  //return *(thread_instance_.get());
  return *(new Caffe());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  //LOG(INFO) << "System entropy source not available, "
  //            "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  //::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  //::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  //::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), root_solver_(true) { }

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  //Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

#else  // Normal GPU + CPU Caffe.

#endif  // CPU_ONLY

}  // namespace caffe
