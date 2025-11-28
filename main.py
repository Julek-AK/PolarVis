
from gui.main_window import run_main_window


from tests.processing_tests.test_case_generator import *
from tests.processing_tests.test_runner import TestRunner, TestBatchRunner
from tests.processing_tests.test_analyzer import visualize_test_result
from processing.torch_backend import analytic_resolve_metapixels_for_testing

from core.file_manager import CacheManager


def run_tests():
    # Prepare some hyperparameters
    img_size = (64, 64)
    seed = 42

    # Initiate generators and runners
    spect_gen = SpectrumGenerator(img_size)
    rand_gen = RandomisedGenerator(img_size)
    const_gen = ConstantGenerator(img_size)

    runner = TestRunner(solver_callable=analytic_resolve_metapixels_for_testing)

    # Spectrum cases
    spect_case1 = spect_gen.generate_case(seed)
    spect_result1 = runner.run_test_case(spect_case1)
    visualize_test_result(spect_result1, "spectrum_case_1")

    # Random cases
    case1 = rand_gen.generate_case(seed)
    case1_result = runner.run_test_case(case1)
    visualize_test_result(case1_result, "random_case_1")

    # Quantized cases
    bit_depths = [16, 8, 4, 2]
    for i, depth in enumerate(bit_depths):
        quant_gen = QuantizingAugmenter(rand_gen, bit_depth=depth)
        quant_case = quant_gen.generate_case(seed)
        quant_result = runner.run_test_case(quant_case)
        visualize_test_result(quant_result, f"quantized_case_{i+1}")

    # Noise cases
    noise_sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    for i, sigma in enumerate(noise_sigmas):
        noise_gen = NoiseAugmenter(const_gen, noise_sigma=sigma)
        noise_case = noise_gen.generate_case(seed)
        noise_result = runner.run_test_case(noise_case)
        visualize_test_result(noise_result, f"noise_case_{i+1}")

    # Time performance
    img_path = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\Honours Programme\Media\Lens Testing\8.png"
    gen = FromImageGenerator(img_path)
    test_case = gen.generate_case()
    runner_cpu = TestRunner(solver_callable=analytic_resolve_metapixels_for_testing, solver_kwargs={"device": 'cpu'})
    runner_cuda = TestRunner(solver_callable=analytic_resolve_metapixels_for_testing, solver_kwargs={"device": 'cuda'})

    cpu_result = runner_cpu.run_test_case(test_case)
    cuda_result = runner_cuda.run_test_case(test_case)

    visualize_test_result(cpu_result, "timed_case_cpu")
    visualize_test_result(cuda_result, "timed_case_cuda")

    # Boundary test cases
    combine_gen = DiagonalCombiner(const_gen, const_gen)
    combine_cases = combine_gen.generate_batch(5, start_seed=42)
    for i, combine_case in enumerate(combine_cases):
        result = runner.run_test_case(combine_case)
        visualize_test_result(result, f"combined_case_{i+1}")
        

if __name__ == '__main__':
    # run_main_window()

    run_tests()


