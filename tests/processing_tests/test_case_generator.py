# Builtins
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image

# External
from numpy.typing import NDArray
import numpy as np
import scipy as scp

# Internal
from core.utils import simulate_image


@dataclass
class TestCase:
    image: NDArray
    ground_truth: Optional[NDArray]
    metadata: Dict[str, Any]


class TestCaseGenerator(ABC):
    """
    Base class for all test case generators
    """
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        """
        Generate test case. If seed is passed, use it for reproducibility
        """
        pass

    def generate_batch(self, n_cases: int, start_seed: Optional[int] = None) -> List[TestCase]:
        """
        Generate a number of test cases. Automatically increment passed seed for reproducibility and variety
        """        
        cases = []
        for i in range(n_cases):
            seed = None if start_seed is None else start_seed + i
            cases.append(self.generate_case(seed=seed))

        return cases

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class FromImageGenerator(TestCaseGenerator):
    """
    Creates a TestCase from an image file, keep in mind such a test case has no ground truth
    """
    def __init__(self, file_path: str) -> None:
        super().__init__("from_image")
        self.file_path = file_path

    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        img = Image.open(self.file_path).convert("L")
        img = np.array(img, dtype=np.float32)

        metadata = {
            "generator": self.name,
            "image_size": img.shape,
            "file_path": self.file_path,
            "seed": seed,
        }

        return TestCase(
            image=img, 
            ground_truth=None, 
            metadata=metadata,
            )
    


class ConstantGenerator(TestCaseGenerator):
    """
    Generator for producing constant polarization fields.
    Can be used deterministically for specified parametrs, or can randomly generate constant fields
    """
    def __init__(self, image_size=(64, 64)) -> None:
        super().__init__("constant")

        self.image_size = image_size
        self.data_size = tuple(x//2 for x in image_size)

    def generate_case_from_params(self, I_pol: float, I_unpol: float, theta: float, seed: Optional[int] = None) -> TestCase:
        data = np.ones((*self.data_size, 3))
        data[..., 0] *= I_pol
        data[..., 1] *= I_unpol
        data[..., 2] *= theta

        img = simulate_image(data)

        metadata = {
            "generator": self.name,
            "image_size": self.image_size,
            "parameters": {
                "I_pol": I_pol,
                "I_unpol": I_unpol,
                "theta": theta,
            },
            "seed": seed
        }

        return TestCase(
            image=img, 
            ground_truth=data, 
            metadata=metadata,
            )
    
    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        rng = np.random.default_rng(seed)

        I_pol   = float(rng.uniform(0.0, 1.0))
        I_unpol = float(rng.uniform(0.0, 1.0))
        theta   = float(rng.uniform(0.0, np.pi))

        return self.generate_case_from_params(I_pol, I_unpol, theta, seed)


class RandomisedGenerator(TestCaseGenerator):
    """
    Generator for producing images from randomly sampled data
    """
    def __init__(self, image_size=(64, 64), noise_type="gaussian") -> None:
        super().__init__("random")
        self.image_size = image_size
        self.data_size = tuple(x//2 for x in image_size)
        self.noise_type = noise_type

    def generate_case(self, seed=None) -> TestCase:
        rng = np.random.default_rng(seed)
        H, W = self.data_size

        I_tot = rng.uniform(0.0, 1.0, (H, W)).astype(np.float32)
        DoLP = rng.uniform(0.0, 1.0, (H, W)).astype(np.float32)

        I_pol   = DoLP * I_tot
        I_unpol = (1.0 - DoLP) * I_tot
        theta = rng.uniform(0.0, np.pi, (H, W)).astype(np.float32)

        data = np.stack([I_pol, I_unpol, theta], axis=-1)
        img = simulate_image(data)

        metadata = {
            "generator": self.name,
            "image_size": self.image_size,
            "seed": seed,
        }

        return TestCase(
            image=img, 
            ground_truth=data, 
            metadata=metadata,
            )
    
class SpectrumGenerator(TestCaseGenerator):
    """
    Generator for constructing a uniformly varying polarization field, with theta changing along x and DoLP along y direction
    """
    def __init__(self, image_size=(64, 64), I_tot=1.0):
        super().__init__("spectrum_theta_dolp")
        self.image_size = image_size
        self.data_size = tuple(x // 2 for x in image_size)
        self.I_tot = I_tot

    def generate_case(self, seed=None) -> TestCase:
        H, W = self.data_size

        theta = np.linspace(0.0, np.pi, W, dtype=np.float32)[None, :].repeat(H, axis=0)
        DoLP  = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

        I_pol   = DoLP * self.I_tot
        I_unpol = (1.0 - DoLP) * self.I_tot

        data = np.stack([I_pol, I_unpol, theta], axis=-1)

        img = simulate_image(data)

        metadata = {
            "generator": self.name,
            "image_size": self.image_size,
            "I_tot": self.I_tot,
            "theta_range": [0, float(np.pi)],
            "DoLP_range": [0, 1],
        }

        return TestCase(
            image=img,
            ground_truth=data,
            metadata=metadata,
        )
    

class BlurAugmenter(TestCaseGenerator):
    """
    Wraps a TestCaseGenerator and adds a specified strength of gaussian blur.
    Ground truth is kept unchanged
    """
    def __init__(self, base_generator: TestCaseGenerator, blur_sigma: float) -> None:
        super().__init__(name=f"{base_generator.name}_blur")

        self.base = base_generator
        self.blur_sigma = blur_sigma

    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        clean_case = self.base.generate_case(seed)

        img = clean_case.image.copy()
        img = scp.ndimage.gaussian_filter(img, sigma=self.blur_sigma, mode='reflect')

        metadata = dict(clean_case.metadata)
        metadata["generator"] = f"{metadata["generator"]}_blur"
        metadata["blur_strength"] = self.blur_sigma

        return TestCase(
            image=img, 
            ground_truth=clean_case.ground_truth, 
            metadata=metadata,
            )


class NoiseAugmenter(TestCaseGenerator):
    """
    Wraps a TestCaseGenerator and adds a specified strength of noise to each pixel.
    Ground truth is kept unchanged
    """
    def __init__(self, base_generator: TestCaseGenerator, noise_sigma: float) -> None:
        super().__init__(name=f"{base_generator.name}_noise")

        self.base = base_generator
        self.noise_sigma = noise_sigma

    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        rng = np.random.default_rng(seed)

        clean_case = self.base.generate_case(seed)

        img = clean_case.image.copy()
        img += rng.normal(0, self.noise_sigma, img.shape)
        img = np.clip(img, 0.0, None)

        metadata = dict(clean_case.metadata)
        metadata["generator"] = f"{metadata["generator"]}_noise"
        metadata["noise_std"] = self.noise_sigma

        return TestCase(
            image=img, 
            ground_truth=clean_case.ground_truth, 
            metadata=metadata,
        )


class QuantizingAugmenter(TestCaseGenerator):
    """
    Wraps a TestCaseGenerator and quantizes the pixel intensities to a specified bit-depth.
    Ground truth is kept unchanged
    """
    def __init__(self, base_generator: TestCaseGenerator, bit_depth: int = 8) -> None:
        super().__init__(f"{base_generator.name}_quantized")

        self.base = base_generator
        self.bit_depth = bit_depth
        self.max_int = (1 << bit_depth) - 1

    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        clean_case = self.base.generate_case(seed)

        img = clean_case.image.copy()

        img = np.clip(img, 0.0, 1.0)
        img_int = np.round(img * self.max_int).astype(np.uint16)
        img_int = np.clip(img_int, 0, self.max_int)

        img_quantized = img_int.astype(np.float32) / self.max_int

        metadata = dict(clean_case.metadata)
        metadata["generator"] = f"{metadata["generator"]}_quantized"
        metadata["bit_depth"] = self.bit_depth

        return TestCase(
            image=img_quantized,
            ground_truth=clean_case.ground_truth,
            metadata=metadata,
        )


class DiagonalCombiner(TestCaseGenerator):
    """
    Combines two TestCaseGenerators by diagonally splicing their images.
    The diagonal runs from top-left to bottom-right unless reversed.
    """

    def __init__(self, gen_A: TestCaseGenerator, gen_B: TestCaseGenerator, reverse: bool = False):
        super().__init__(name=f"{gen_A.name}_X_{gen_B.name}_diag")
        self.gen_A = gen_A
        self.gen_B = gen_B
        self.reverse = reverse

    def generate_case(self, seed: Optional[int] = None) -> TestCase:
        # Get the two cases
        seed_A = seed
        seed_B = None if seed is None else seed*2

        case_A = self.gen_A.generate_case(seed_A)
        case_B = self.gen_B.generate_case(seed_B)

        img_A = case_A.image
        img_B = case_B.image

        assert img_A.shape == img_B.shape
        H, W = img_A.shape[:2]

        # Create diagonal mask
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        mask = yy <= xx  
        if self.reverse:
            mask = ~mask

        # Expand for channels
        if img_A.ndim == 3:
            mask3 = mask[..., None]
        else:
            mask3 = mask

        # Combine images
        combined_img = np.where(mask3, img_A, img_B)

        # Combine ground truths if available
        if case_A.ground_truth is not None and case_B.ground_truth is not None:
            gt_mask = mask[::2, ::2]
            gt_mask3 = gt_mask[..., None]

            combined_gt = np.where(gt_mask3, 
                                   case_A.ground_truth, 
                                   case_B.ground_truth)
        else:
            combined_gt = None

        # Metadata
        metadata = {
            "generator": f"diag({case_A.metadata.get('generator')}, {case_B.metadata.get('generator')})",
            "gen A data": case_A.metadata,
            "gen B data": case_B.metadata,
            "reverse": self.reverse,
        }

        return TestCase(
            image=combined_img,
            ground_truth=combined_gt,
            metadata=metadata,
        )