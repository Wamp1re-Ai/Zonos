import math
from functools import cache
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

from zonos.utils import DEFAULT_DEVICE


class logFbankCal(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ):
        super().__init__()
        self.fbankCal = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            n_mels=n_mels,
        )

    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out


class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, in_ch=1, feat_dim="2d", **kwargs):
        super(ResNet, self).__init__()
        if feat_dim == "1d":
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim == "2d":
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim == "3d":
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            print("error")

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, block_id=3)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, block_id=4)

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride, block_id))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNet293(in_planes: int, **kwargs):
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
    def __init__(
        self,
        in_planes: int = 64,
        embd_dim: int = 256,
        acoustic_dim: int = 80,
        featCal=None,
        dropout: float = 0,
        **kwargs,
    ):
        super(ResNet293_based, self).__init__()
        self.featCal = featCal
        self.front = ResNet293(in_planes)
        block_expansion = SimAMBasicBlock.expansion
        self.pooling = ASP(in_planes * block_expansion, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # Removed
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C, featCal):
        super(ECAPA_TDNN, self).__init__()
        self.featCal = featCal
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Added
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        x = self.featCal(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


class SpeakerEmbedding(nn.Module):
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        with torch.device(device):
            self.model = ResNet293_based()
            state_dict = torch.load(ckpt_path, weights_only=True, mmap=True, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.featCal = logFbankCal()

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, 16_000).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = self._get_resampler(sample_rate)(wav)
        return wav

    def forward(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate).to(self.device, self.dtype)
        return self.model(wav).to(wav.device)


class SpeakerEmbeddingLDA(nn.Module):
    def __init__(self, device: str = DEFAULT_DEVICE):
        super().__init__()
        spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base.pt",
        )
        lda_spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base_LDA-128.pt",
        )

        self.device = device
        with torch.device(device):
            self.model = SpeakerEmbedding(spk_model_path, device)
            lda_sd = torch.load(lda_spk_model_path, weights_only=True)
            out_features, in_features = lda_sd["weight"].shape
            self.lda = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
            self.lda.load_state_dict(lda_sd)

        self.requires_grad_(False).eval()

    def forward(self, wav: torch.Tensor, sample_rate: int):
        emb = self.model(wav, sample_rate).to(torch.float32)
        return emb, self.lda(emb)


# Enhanced Voice Cloning Functions
def preprocess_audio_for_cloning(wav: torch.Tensor, sample_rate: int,
                                target_length_seconds: float = None,
                                normalize: bool = True,
                                remove_silence: bool = True) -> torch.Tensor:
    """
    Preprocess audio for better voice cloning quality.

    Args:
        wav: Input audio tensor
        sample_rate: Sample rate of the audio
        target_length_seconds: Target length in seconds (None for no trimming)
        normalize: Whether to normalize audio amplitude
        remove_silence: Whether to remove leading/trailing silence

    Returns:
        Preprocessed audio tensor
    """
    # Convert to mono if stereo
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Remove silence from beginning and end
    if remove_silence:
        # Simple energy-based silence detection
        energy = wav.pow(2).mean(dim=0)
        threshold = energy.max() * 0.01  # 1% of max energy
        non_silent = energy > threshold

        if non_silent.any():
            start_idx = non_silent.nonzero()[0].item()
            end_idx = non_silent.nonzero()[-1].item() + 1
            wav = wav[:, start_idx:end_idx]

    # Normalize audio
    if normalize:
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav / max_val * 0.95  # Normalize to 95% to avoid clipping

    # Trim to target length if specified
    if target_length_seconds is not None:
        target_samples = int(target_length_seconds * sample_rate)
        if wav.shape[1] > target_samples:
            # Take from the middle for better voice characteristics
            start_idx = (wav.shape[1] - target_samples) // 2
            wav = wav[:, start_idx:start_idx + target_samples]
        elif wav.shape[1] < target_samples:
            # Pad with silence if too short
            padding = target_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, padding))

    return wav


def analyze_voice_quality(wav: torch.Tensor, sample_rate: int) -> dict:
    """
    Analyze voice quality metrics for cloning assessment.

    Args:
        wav: Audio tensor
        sample_rate: Sample rate

    Returns:
        Dictionary with quality metrics
    """
    # Convert to mono if needed
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(0)
    elif wav.ndim == 1:
        pass
    else:
        wav = wav.squeeze()

    # Basic quality metrics
    duration = wav.shape[0] / sample_rate
    rms_energy = torch.sqrt(torch.mean(wav**2)).item()
    peak_amplitude = torch.max(torch.abs(wav)).item()

    # Signal-to-noise ratio estimation (simple)
    # Use the quietest 10% as noise estimate
    sorted_abs = torch.sort(torch.abs(wav))[0]
    noise_level = sorted_abs[:len(sorted_abs)//10].mean().item()
    signal_level = rms_energy
    snr_estimate = 20 * torch.log10(torch.tensor(signal_level / (noise_level + 1e-8))).item()

    # Dynamic range
    dynamic_range = 20 * torch.log10(torch.tensor(peak_amplitude / (rms_energy + 1e-8))).item()

    return {
        'duration': duration,
        'rms_energy': rms_energy,
        'peak_amplitude': peak_amplitude,
        'snr_estimate': snr_estimate,
        'dynamic_range': dynamic_range,
        'quality_score': min(1.0, max(0.0, (snr_estimate + 20) / 40))  # Normalized quality score
    }


def get_voice_cloning_conditioning_params(voice_quality: dict = None) -> dict:
    """
    Get optimized conditioning parameters for voice cloning based on voice quality.

    Args:
        voice_quality: Voice quality metrics from analyze_voice_quality

    Returns:
        Dictionary of conditioning parameters optimized for voice cloning
    """
    # Base parameters optimized for voice cloning
    base_params = {
        'emotion': [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.5],  # More neutral, less emotional variation
        'fmax': 22050.0,  # Optimal for voice cloning
        'pitch_std': 15.0,  # Reduced for more consistent pitch
        'speaking_rate': 12.0,  # Slower, more controlled rate
        'vqscore_8': [0.75] * 8,  # Slightly lower for more natural sound
        'dnsmos_ovrl': 3.8,  # Slightly lower for naturalness
        'speaker_noised': False,  # Clean speaker embedding
    }

    # Adjust based on voice quality if provided
    if voice_quality:
        quality_score = voice_quality.get('quality_score', 0.5)

        # Adjust pitch variation based on quality
        if quality_score > 0.7:
            base_params['pitch_std'] = 18.0  # Can handle slightly more variation
        elif quality_score < 0.3:
            base_params['pitch_std'] = 12.0  # Very conservative for poor quality

        # Adjust speaking rate based on quality
        if quality_score > 0.8:
            base_params['speaking_rate'] = 14.0  # Can be slightly faster
        elif quality_score < 0.4:
            base_params['speaking_rate'] = 10.0  # Very slow for poor quality

        # Adjust VQ score based on quality
        if quality_score > 0.6:
            base_params['vqscore_8'] = [0.78] * 8
        else:
            base_params['vqscore_8'] = [0.72] * 8

    return base_params


def get_voice_cloning_sampling_params(voice_quality: dict = None) -> dict:
    """
    Get optimized sampling parameters for voice cloning.

    Args:
        voice_quality: Voice quality metrics from analyze_voice_quality

    Returns:
        Dictionary of sampling parameters optimized for voice cloning
    """
    # Conservative sampling for consistency
    base_params = {
        'min_p': 0.05,  # More conservative than default 0.1
        'top_k': 0,     # Disabled
        'top_p': 0.0,   # Disabled
        'temperature': 0.8,  # Slightly lower temperature for consistency
        'repetition_penalty': 1.5,  # Lower repetition penalty to avoid unnatural pauses
        'repetition_penalty_window': 3,  # Shorter window
    }

    # Adjust based on voice quality
    if voice_quality:
        quality_score = voice_quality.get('quality_score', 0.5)

        if quality_score > 0.7:
            # High quality voice can handle slightly more variation
            base_params['min_p'] = 0.08
            base_params['temperature'] = 0.85
        elif quality_score < 0.4:
            # Poor quality voice needs very conservative sampling
            base_params['min_p'] = 0.03
            base_params['temperature'] = 0.7
            base_params['repetition_penalty'] = 1.2

    return base_params
