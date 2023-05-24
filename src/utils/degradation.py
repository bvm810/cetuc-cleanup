import torch
from scipy.signal import firwin
from torch.nn.functional import conv1d
import matplotlib.pyplot as plt


class Degradation:
    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        "Assumes signal shape to be (batch_size, audio_length)"
        raise NotImplementedError


class GaussianNoise(Degradation):
    def __init__(self, mean: float, variance: float) -> None:
        self.mean = mean
        self.variance = variance

    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        noise = self.mean + torch.randn(signal.size()) * self.variance
        return signal + noise


class WhiteGaussianNoise(GaussianNoise):
    def __init__(self, variance: float) -> None:
        super().__init__(0, variance)


class ColoredGaussianNoise(GaussianNoise):
    def __init__(self, variance: float, filter_args: dict) -> None:
        super().__init__(0, variance)
        self.filter = torch.Tensor(firwin(**filter_args))
        self.filter = self.filter.view(1, 1, -1)

    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        white_noise = torch.randn(signal.size()) * self.variance
        zero_pad = torch.zeros((signal.size(0), self.filter.size(-1) - 1))
        white_noise = torch.cat((white_noise, zero_pad), dim=1)
        white_noise = torch.unsqueeze(white_noise, 1)
        colored_noise = conv1d(white_noise, self.filter).squeeze()
        return signal + colored_noise


class MemorylessPolynomial(Degradation):
    """
    Pointwise degradation with polynomial function of each sample
    y[n] = coefs[0] * x[n] + coefs[1] * (x[n] ** 2) + ... + coefs[order-1] * (x[n] ** order)
    """

    def __init__(self, coefs: torch.Tensor) -> None:
        self.coefs = coefs
        self.exponents = torch.arange(1, coefs.size(-1) + 1)

    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.unsqueeze(-1)  # shape (bs, len, 1)
        exps = self.exponents.view(1, 1, -1)  # shape (1, 1, order)
        powers = torch.pow(signal, exps)  # shape (bs, len, order)
        degraded_signal = torch.matmul(powers, self.coefs.view(-1, 1))
        return degraded_signal.squeeze(-1)


class Gap(Degradation):
    def __init__(self, start_sample: int, length_in_samples: int) -> None:
        self.length = length_in_samples
        self.start = start_sample

    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        if self.start + self.length >= signal.size(-1):
            raise ValueError("Gap exceeds signal size")
        first_part = signal[:, : self.start]
        zeros = torch.zeros((signal.size(0), self.length))
        second_part = signal[:, self.start + self.length :]
        return torch.cat((first_part, zeros, second_part), 1)


class Clip(Degradation):
    def __init__(self, min: float, max: float) -> None:
        if min >= max:
            raise ValueError("Clip minimum value exceeds clip max value")
        self.min = min
        self.max = max

    def degrade(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.clip(signal, min=self.min, max=self.max)


class SymmetricalClip(Clip):
    def __init__(self, clip_value: float) -> None:
        super().__init__(-clip_value, clip_value)


if __name__ == "__main__":
    # example inputs
    Nfft = 256
    fs = 1000
    zero = torch.zeros((1, Nfft))
    frequencies = torch.arange(0, Nfft // 2 + 1) * (fs / Nfft)
    samples = torch.arange(0, Nfft)
    sine = torch.cos(2 * torch.pi * (samples / (Nfft // 2))).unsqueeze(0)

    # white gaussian noise
    # wgn = WhiteGaussianNoise(variance=1)
    # white_noise = wgn.degrade(zero).squeeze()
    # plt.figure()
    # white_noise_spectrum = torch.abs(torch.fft.rfft(white_noise, n=Nfft))
    # plt.plot(frequencies, white_noise_spectrum)
    # plt.show()

    # low frequency gaussian noise
    # filter_args = {"numtaps": 20, "cutoff": 200, "width": 50, "fs": fs}
    # cgn = ColoredGaussianNoise(variance=1, filter_args=filter_args)
    # colored_noise = cgn.degrade(zero).squeeze()
    # plt.figure()
    # colored_noise_spectrum = torch.abs(torch.fft.rfft(colored_noise, n=Nfft))
    # plt.plot(frequencies, colored_noise_spectrum)
    # plt.show()

    # band-limited gaussian noise
    filter_args = {
        "numtaps": 20,
        "cutoff": [100, 200],
        "width": 50,
        "pass_zero": False,
        "fs": fs,
    }
    cgn = ColoredGaussianNoise(variance=1, filter_args=filter_args)
    colored_noise = cgn.degrade(zero).squeeze()
    plt.figure()
    colored_noise_spectrum = torch.abs(torch.fft.rfft(colored_noise, n=Nfft))
    plt.plot(frequencies, colored_noise_spectrum)
    plt.show()

    # gap
    # gap = Gap(50, 50)
    # gapped_sine = gap.degrade(sine).squeeze()
    # plt.figure()
    # plt.plot(samples, gapped_sine)
    # plt.show()

    # clip
    # clip = Clip(-0.5, 0.25)
    # clipped_sine = clip.degrade(sine).squeeze()
    # plt.figure()
    # plt.plot(samples, clipped_sine)
    # plt.show()

    # poly distortion
    # coefs = torch.Tensor([3, 2, 1])
    # poly = MemorylessPolynomial(coefs)
    # poly_sine = poly.degrade(sine).squeeze()
    # plt.figure()
    # plt.plot(samples, poly_sine)
    # plt.show()
