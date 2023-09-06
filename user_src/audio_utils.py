import io
from pathlib import Path

import numpy as np

from fairseq.data.audio.audio_utils import (
    FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS,
    get_fbank,
    get_waveform,
    is_npy_data,
    is_sf_audio_data,
    parse_path,
    read_from_stored_zip,
)


def get_features_from_npy_or_audio(path):
    ext = Path(path).suffix
    if ext not in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_stored_zip(
    path,
    byte_offset,
    byte_size,
    need_waveform=False,
    use_sample_rate=None,
):
    assert path.endswith(".zip")
    data = read_from_stored_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_sf_audio_data(data):
        features_or_waveform = (
            get_waveform(f, always_2d=False, output_sample_rate=use_sample_rate)[0]
            if need_waveform
            else get_fbank(f)
        )
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_features_or_waveform(path: str, need_waveform=False, use_sample_rate=None):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    if len(slice_ptr) == 0:
        if need_waveform:
            return get_waveform(
                _path, always_2d=False, output_sample_rate=use_sample_rate
            )[0]
        return get_features_from_npy_or_audio(_path)
    elif len(slice_ptr) == 2:
        features_or_waveform = get_features_or_waveform_from_stored_zip(
            _path,
            slice_ptr[0],
            slice_ptr[1],
            need_waveform=need_waveform,
            use_sample_rate=use_sample_rate,
        )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform
