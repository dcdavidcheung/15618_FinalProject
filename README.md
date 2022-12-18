# TITLE: Multiple Method Parallel Raytracing

David Cheung (dcheung2)

Shane Li (sihengl)


## URL: https://github.com/dcdavidcheung/15618_FinalProject


## SUMMARY: 

We will be speeding up ray tracing with different parallel computing models and seeing how the performance in speedup changes with each model.


## BACKGROUND:

In raytracing, we shoot out a ray through the center of each pixel of the screen and calculate the color of that pixel by accumulating colors from the albedos. This operation is not too expensive by itself, but the amount of times we have to repeat it makes the computation time grow exceptionally fast. Firstly, there can be thousands of pixels on a canvas. A standard 1080x1920 pixel screen has over 2 million pixels, each of which will require a raytracing operation to fill in. On top of that, taking only one sample per pixel will give a horribly noisy image. Since raytracing relies on Monte Carlo integration, we must take a large amount of samples per pixel in order for the pixel color to converge towards the true value. Even then, we might get very noisy images based on the behavior of the scene. In particular, specular surfaces are difficult to render due to their property of reflecting light in very specific ways.

Given all of this, raytracing benefits greatly from any sort of acceleration, as each little thing we optimize will give significant speedup due to how many times the operations are repeated. More importantly, each raytracing operation is independent from one another. This leads to raytracing being an easy target for parallelism. Some challenges might arise with parallelization techniques though, and we plan on exploring these different techniques and their tradeoffs.


## THE CHALLENGE: 

There are a lot of benefits that can be gained from exploiting locality since there are a lot of objects in some of the scenes. The most significant challenge will be translating the code to fit into the different parallel computing models which may cause new dependencies and bottlenecks to show up. For raytracing, how to use some models of computation is also quite unclear. For example, given how much variance there is in the computation of each pixel, SIMD will likely achieve little to no benefit. Not only that, in large scenes, the amount of objects we have to do intersection tests against will be very large, which can lead to poor spatial and temporal locality if mismanaged. However, we want to try to get a speedup of at least multiple times faster than the baseline.


## RESOURCES:

We will be taking code from a 15-668 assignment as our starter code. We will mostly develop on the gates machines and try to benchmark performance on the Pittsburgh Supercomputer Cluster. 


## GOALS AND DELIVERABLES:

We plan to be able to parallelize the code with OpenMP, MPI, and SIMD instructions. For each parallelization model, we plan to benchmark the performance with increasingly complex scenes to gauge performance and bottlenecks associated with each programming model. As for stretch goals, we currently are thinking about implementing task queues and/or building data structures that can more effectively capture the inherent spatial locality. We can also look into changing the architecture and try parallelizing with CUDA. We will place more of an emphasis on the SIMD implementation over the CUDA version so that we will switch architectures only if we have time to. 

Our Final demo will essentially be a display of the speedup graphs taken from the benchmarks. This will be augmented with graphs depicting where each programming model’s bottlenecks lie.


## PLATFORM CHOICE:

We plan on running on multicore CPUs and working with C++. OpenMP, MPI, and SIMD all have support with modern C++ compilers. 



## SCHEDULE: 

- 11/14-11/18  
	- David: Start implementing MPI version  
	- Shane: Start implementing OpenMP version  
- 11/21-11/25
	- David: Finish implementing MPI version  
	- Shane: Finish implementing OpenMP version  
    - David and Shane: Test for correctness and benchmark MPI and OpenMP versions  
- 11/28-12/2  
	- David and Shane: Write and submit intermediate milestone update  
    - David: Start load balancing MPI version  
    - Shane: Start implementing SIMD version  
- 12/5-12/9  
    - David: Finish load balancing MPI version  
    - Shane: Finish implementing SIMD version  
    - David and Shane: Test for correctness and benchmark and SIMD versions  
- 12/12-12/16  
    - David and Shane: Benchmark with more complex scenes  
	- David and Shane: Finalize final writeup and prepare demo  
  
Final report found in `final_report.pdf`